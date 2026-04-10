import type { IncomingMessage, ServerResponse } from "http";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { env } from "../core/cfg";
import {
    add_hsg_memory,
    hsg_query,
    reinforce_memory,
    delete_memory,
    sector_configs,
} from "../memory/hsg";
import { q, all_async, memories_table, vector_store } from "../core/db";
import { getEmbeddingInfo } from "../memory/embed";
import { j, p } from "../utils";
import type { sector_type, mem_row, rpc_err_code } from "../core/types";
import { update_user_summary } from "../memory/user_summary";
import { insert_fact } from "../temporal_graph/store";
import { query_facts_at_time } from "../temporal_graph/query";
import { ToolRegistry } from "./mcp_tools";

const sec_enum = z.enum([
    "episodic",
    "semantic",
    "procedural",
    "emotional",
    "reflective",
] as const);

const trunc = (val: string, max = 200) =>
    val.length <= max ? val : `${val.slice(0, max).trimEnd()}...`;

const build_mem_snap = (row: mem_row) => ({
    id: row.id,
    primary_sector: row.primary_sector,
    salience: Number(row.salience.toFixed(3)),
    last_seen_at: row.last_seen_at,
    user_id: row.user_id,
    content_preview: trunc(row.content, 240),
});

const fmt_matches = (matches: Awaited<ReturnType<typeof hsg_query>>) =>
    matches
        .map((m: any, idx: any) => {
            const prev = trunc(m.content.replace(/\s+/g, " ").trim(), 200);
            return `${idx + 1}. [${m.primary_sector}] score=${m.score.toFixed(3)} salience=${m.salience.toFixed(3)} id=${m.id}\n${prev}`;
        })
        .join("\n\n");

const set_hdrs = (res: ServerResponse) => {
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "POST,OPTIONS");
    res.setHeader(
        "Access-Control-Allow-Headers",
        "Content-Type,Authorization,Mcp-Session-Id",
    );
};

const send_err = (
    res: ServerResponse,
    code: rpc_err_code,
    msg: string,
    id: number | string | null = null,
    status = 400,
) => {
    if (!res.headersSent) {
        res.statusCode = status;
        set_hdrs(res);
        res.end(
            JSON.stringify({
                jsonrpc: "2.0",
                error: { code, message: msg },
                id,
            }),
        );
    }
};

const uid = (val?: string | null) => (val?.trim() ? val.trim() : undefined);

export const create_mcp_srv = () => {
    const srv = new McpServer(
        {
            name: "openmemory-mcp",
            version: "2.1.0",
        },
        { capabilities: { tools: {}, resources: {}, logging: {} } },
    );

    const registry = new ToolRegistry();

    registry.tool(
        "openmemory_query",
        "Query OpenMemory for contextual memories (HSG) and/or temporal facts",
        {
            query: z
                .string()
                .min(1, "query text is required")
                .describe("Free-form search text"),
            type: z
                .enum(["contextual", "factual", "unified"])
                .optional()
                .default("contextual")
                .describe(
                    "Query type: 'contextual' for HSG semantic search (default), 'factual' for temporal fact queries, 'unified' for both",
                ),
            fact_pattern: z
                .object({
                    subject: z
                        .string()
                        .optional()
                        .describe("Subject pattern (entity) - use undefined for wildcard"),
                    predicate: z
                        .string()
                        .optional()
                        .describe("Predicate pattern (relationship) - use undefined for wildcard"),
                    object: z
                        .string()
                        .optional()
                        .describe("Object pattern (value) - use undefined for wildcard"),
                })
                .optional()
                .describe(
                    "Fact pattern for temporal queries. Used when type is 'factual' or 'unified'",
                ),
            at: z
                .string()
                .optional()
                .describe(
                    "ISO date string for point-in-time queries (default: now). Queries facts valid at this time",
                ),
            k: z
                .number()
                .int()
                .min(1)
                .max(32)
                .default(8)
                .describe("Maximum results to return (for HSG queries)"),
            sector: sec_enum
                .optional()
                .describe("Restrict search to a specific sector (for HSG queries)"),
            min_salience: z
                .number()
                .min(0)
                .max(1)
                .optional()
                .describe("Minimum salience threshold (for HSG queries)"),
            min_score: z
                .number()
                .min(0)
                .max(1)
                .optional()
                .describe("Minimum score threshold to filter low-relevance results"),
            user_id: z
                .string()
                .trim()
                .min(1)
                .optional()
                .describe("Isolate results to a specific user identifier"),
            project: z
                .string()
                .trim()
                .min(1)
                .optional()
                .describe("Restrict search to a specific project scope"),
        },
        async ({
            query,
            type = "contextual",
            fact_pattern,
            at,
            k,
            sector,
            min_salience,
            min_score,
            user_id,
            project,
        }) => {
            const u = uid(user_id);
            const proj = project?.trim() || undefined;
            const results: any = { type, query };
            const at_date = at ? new Date(at) : new Date();


            if (type === "contextual" || type === "unified") {
                const flt =
                    sector || min_salience !== undefined || u || proj
                        ? {
                            ...(sector ? { sectors: [sector as sector_type] } : {}),
                            ...(min_salience !== undefined
                                ? { minSalience: min_salience }
                                : {}),
                            ...(u ? { user_id: u } : {}),
                            ...(proj ? { project: proj } : {}),
                        }
                        : undefined;

                let matches = await hsg_query(query, k ?? 8, flt);

                // BM25 hybrid: boost vector results that also match FTS5
                if (matches.length > 0) {
                    try {
                        const fts_hits = await q.fts_search.all(query, 50);
                        if (fts_hits.length > 0) {
                            const fts_ids = new Set(fts_hits.map((h: any) => h.id));
                            for (const m of matches) {
                                if (fts_ids.has(m.id)) {
                                    (m as any).score = Math.min(1.0, m.score * 1.15);
                                }
                            }
                            matches.sort((a: any, b: any) => b.score - a.score);
                        }
                    } catch { /* FTS5 not available (Postgres), skip */ }
                }

                if (min_score !== undefined) {
                    matches = matches.filter((m: any) => m.score >= min_score);
                }

                results.contextual = matches.map((m: any) => ({
                    source: "hsg",
                    id: m.id,
                    score: Number(m.score.toFixed(4)),
                    primary_sector: m.primary_sector,
                    sectors: m.sectors,
                    salience: Number(m.salience.toFixed(4)),
                    last_seen_at: m.last_seen_at,
                    path: m.path,
                    content: m.content,
                }));
            }


            if (type === "factual" || type === "unified") {
                const facts = await query_facts_at_time(
                    fact_pattern?.subject,
                    fact_pattern?.predicate,
                    fact_pattern?.object,
                    at_date,
                    0.0,
                    u,
                );

                results.factual = facts.map((f: any) => ({
                    source: "temporal",
                    id: f.id,
                    subject: f.subject,
                    predicate: f.predicate,
                    object: f.object,
                    valid_from: f.valid_from,
                    valid_to: f.valid_to,
                    confidence: Number(f.confidence.toFixed(4)),
                    content: `${f.subject} ${f.predicate} ${f.object}`,
                }));
            }


            let summ = "";
            if (type === "contextual") {
                summ = results.contextual.length
                    ? fmt_matches(results.contextual)
                    : "No contextual memories matched the query.";
            } else if (type === "factual") {
                if (results.factual.length === 0) {
                    summ = "No temporal facts matched the query.";
                } else {
                    summ = results.factual
                        .map(
                            (f: any, idx: number) =>
                                `${idx + 1}. [fact] confidence=${f.confidence} id=${f.id}\n${f.content}`,
                        )
                        .join("\n\n");
                }
            } else {

                const ctx_count = results.contextual?.length || 0;
                const fact_count = results.factual?.length || 0;
                summ = `Found ${ctx_count} contextual memories and ${fact_count} temporal facts.\n\n`;

                if (ctx_count > 0) {
                    summ += "=== Contextual Memories ===\n";
                    summ += fmt_matches(results.contextual) + "\n\n";
                }

                if (fact_count > 0) {
                    summ += "=== Temporal Facts ===\n";
                    summ += results.factual
                        .map(
                            (f: any, idx: number) =>
                                `${idx + 1}. [fact] confidence=${f.confidence}\n${f.content}`,
                        )
                        .join("\n\n");
                }

                if (ctx_count === 0 && fact_count === 0) {
                    summ = "No results found in either system.";
                }
            }

            return {
                content: [
                    { type: "text", text: summ },
                    {
                        type: "text",
                        text: JSON.stringify(results, null, 2),
                    },
                ],
            };
        },
    );

    registry.tool(
        "openmemory_store",
        "Persist new content into OpenMemory (HSG contextual memory and/or temporal facts)",
        {
            content: z.string().min(1).describe("Raw memory text to store"),
            type: z
                .enum(["contextual", "factual", "both"])
                .optional()
                .default("contextual")
                .describe(
                    "Storage type: 'contextual' for HSG only (default), 'factual' for temporal facts only, 'both' for both systems",
                ),
            facts: z
                .array(
                    z.object({
                        subject: z.string().min(1).describe("Fact subject (entity)"),
                        predicate: z
                            .string()
                            .min(1)
                            .describe("Fact predicate (relationship)"),
                        object: z.string().min(1).describe("Fact object (value)"),
                        confidence: z
                            .number()
                            .min(0)
                            .max(1)
                            .optional()
                            .describe("Confidence score (0-1, default 1.0)"),
                        valid_from: z
                            .string()
                            .optional()
                            .describe(
                                "ISO date string for fact validity start (default: now)",
                            ),
                    }),
                )
                .optional()
                .describe(
                    "Array of facts to store in temporal graph. Required when type is 'factual' or 'both'",
                ),
            tags: z
                .array(z.string())
                .optional()
                .describe("Optional tag list (for HSG storage)"),
            metadata: z
                .record(z.any())
                .optional()
                .describe("Arbitrary metadata blob"),
            user_id: z
                .string()
                .trim()
                .min(1)
                .optional()
                .describe(
                    "Associate the memory with a specific user identifier",
                ),
            project: z
                .string()
                .trim()
                .min(1)
                .optional()
                .describe(
                    "Project scope for this memory (e.g., directory basename)",
                ),
            session_id: z
                .string()
                .trim()
                .min(1)
                .optional()
                .describe("Session identifier for grouping related memories"),
            observation_type: z
                .enum([
                    "observation",
                    "bugfix",
                    "decision",
                    "discovery",
                    "feature",
                    "gotcha",
                    "refactor",
                ])
                .optional()
                .default("observation")
                .describe("Type of observation being stored"),
        },
        async ({
            content,
            type = "contextual",
            facts,
            tags,
            metadata,
            user_id,
            project,
            session_id,
            observation_type,
        }) => {
            const u = uid(user_id);
            const results: any = { type };


            if (
                (type === "factual" || type === "both") &&
                (!facts || facts.length === 0)
            ) {
                throw new Error(
                    `Facts array is required when type is '${type}'. Please provide at least one fact.`,
                );
            }


            if (type === "contextual" || type === "both") {
                const res = await add_hsg_memory(
                    content,
                    j(tags || []),
                    metadata,
                    u,
                    project,
                    session_id,
                    observation_type,
                );
                results.hsg = {
                    id: res.id,
                    primary_sector: res.primary_sector,
                    sectors: res.sectors,
                };

                if (u) {
                    update_user_summary(u).catch((err) =>
                        console.error("[MCP] user summary update failed:", err),
                    );
                }
            }


            if ((type === "factual" || type === "both") && facts) {
                const temporal_results = [];
                for (const fact of facts) {
                    const valid_from = fact.valid_from
                        ? new Date(fact.valid_from)
                        : new Date();
                    const confidence = fact.confidence ?? 1.0;

                    const fact_id = await insert_fact(
                        fact.subject,
                        fact.predicate,
                        fact.object,
                        valid_from,
                        confidence,
                        metadata,
                        u,
                    );

                    temporal_results.push({
                        id: fact_id,
                        subject: fact.subject,
                        predicate: fact.predicate,
                        object: fact.object,
                        valid_from: valid_from.toISOString(),
                        confidence,
                    });
                }
                results.temporal = temporal_results;
            }


            let txt = "";
            if (type === "contextual") {
                txt = `Stored memory ${results.hsg.id} (primary=${results.hsg.primary_sector}) across sectors: ${results.hsg.sectors.join(", ")}${u ? ` [user=${u}]` : ""}`;
            } else if (type === "factual") {
                txt = `Stored ${results.temporal.length} temporal fact(s)${u ? ` [user=${u}]` : ""}`;
            } else {
                txt = `Stored in both systems: HSG memory ${results.hsg.id} + ${results.temporal.length} temporal fact(s)${u ? ` [user=${u}]` : ""}`;
            }

            return {
                content: [
                    { type: "text", text: txt },
                    {
                        type: "text",
                        text: JSON.stringify(
                            { ...results, user_id: u ?? null },
                            null,
                            2,
                        ),
                    },
                ],
            };
        },
    );

    registry.tool(
        "openmemory_reinforce",
        "Boost salience for an existing memory",
        {
            id: z.string().min(1).describe("Memory identifier to reinforce"),
            boost: z
                .number()
                .min(0.01)
                .max(1)
                .default(0.1)
                .describe("Salience boost amount (default 0.1)"),
        },
        async ({ id, boost }) => {
            await reinforce_memory(id, boost);
            return {
                content: [
                    {
                        type: "text",
                        text: `Reinforced memory ${id} by ${boost}`,
                    },
                ],
            };
        },
    );

    registry.tool(
        "openmemory_delete",
        "Delete a memory by identifier",
        {
            id: z.string().min(1).describe("Memory identifier to delete"),
            user_id: z.string().trim().min(1).optional().describe("Validate ownership"),
        },
        async ({ id, user_id }) => {
            const u = uid(user_id);
            if (u) {
                // Pre-check ownership if user_id provided
                const mem = await q.get_mem.get(id);
                if (mem && mem.user_id !== u) {
                    throw new Error(`Memory ${id} not found for user ${u}`);
                }
            }

            const success = await delete_memory(id);
            if (!success) {
                return {
                    content: [{ type: "text", text: `Memory ${id} not found or could not be deleted.` }],
                    isError: true
                };
            }

            return {
                content: [{ type: "text", text: `Memory ${id} successfully deleted.` }],
            };
        },
    );

    registry.tool(
        "openmemory_list",
        "List recent memories for quick inspection",
        {
            limit: z
                .number()
                .int()
                .min(1)
                .max(50)
                .default(10)
                .describe("Number of memories to return"),
            sector: sec_enum.optional().describe("Optionally limit to a sector"),
            user_id: z
                .string()
                .trim()
                .min(1)
                .optional()
                .describe("Restrict results to a specific user identifier"),
            project: z
                .string()
                .trim()
                .min(1)
                .optional()
                .describe("Restrict to a specific project"),
        },
        async ({ limit, sector, user_id, project }) => {
            const u = uid(user_id);
            const proj = project?.trim() || undefined;
            let rows: mem_row[];
            if (u) {
                const all = await q.all_mem_by_user.all(u, limit ?? 10, 0);
                rows = all.filter(
                    (row) =>
                        (!sector || row.primary_sector === sector) &&
                        (!proj || (row as any).project === proj),
                );
            } else {
                if (proj) {
                    const scoped = await q.all_mem_by_project.all(
                        proj,
                        limit ?? 10,
                        0,
                    );
                    rows = sector
                        ? scoped.filter((row) => row.primary_sector === sector)
                        : scoped;
                } else {
                    rows = sector
                        ? await q.all_mem_by_sector.all(sector, limit ?? 10, 0)
                        : await q.all_mem.all(limit ?? 10, 0);
                }
            }
            const items = rows.map((row) => ({
                ...build_mem_snap(row),
                tags: p(row.tags || "[]") as string[],
                metadata: p(row.meta || "{}") as Record<string, unknown>,
            }));
            const lns = items.map(
                (item, idx) =>
                    `${idx + 1}. [${item.primary_sector}] salience=${item.salience} id=${item.id}${item.tags.length ? ` tags=${item.tags.join(", ")}` : ""}${item.user_id ? ` user=${item.user_id}` : ""}\n${item.content_preview}`,
            );
            return {
                content: [
                    {
                        type: "text",
                        text: lns.join("\n\n") || "No memories stored yet.",
                    },
                    { type: "text", text: JSON.stringify({ items }, null, 2) },
                ],
            };
        },
    );

    registry.tool(
        "openmemory_summarize",
        "Persist a summary of work completed in a session.",
        {
            session_id: z.string().min(1).describe("Session identifier"),
            project: z
                .string()
                .min(1)
                .describe("Project name (working directory basename)"),
            request: z.string().min(1).describe("What the user originally asked for"),
            completed: z.string().min(1).describe("What was actually completed"),
            learned: z.string().min(1).describe("Key learnings and discoveries"),
            next_steps: z.string().optional().describe("Remaining work or follow-ups"),
            files_modified: z
                .array(z.string())
                .optional()
                .describe("List of modified file paths"),
        },
        async (args) => {
            const existing = await q.get_session.get(args.session_id);
            if (!existing) {
                await q.ins_session.run(
                    args.session_id,
                    args.project,
                    Date.now(),
                    null,
                );
            }
            await q.ins_summary.run(
                args.session_id,
                args.project,
                args.request,
                args.completed,
                args.learned,
                args.next_steps || null,
                args.files_modified ? JSON.stringify(args.files_modified) : null,
                Date.now(),
            );
            return {
                content: [
                    {
                        type: "text",
                        text: `Session summary saved for project "${args.project}"`,
                    },
                ],
            };
        },
    );

    registry.tool(
        "openmemory_timeline",
        "Get surrounding observations around an anchor memory for chronological context.",
        {
            memory_id: z.string().min(1).describe("Anchor memory ID"),
            depth_before: z
                .number()
                .int()
                .min(0)
                .max(50)
                .default(3)
                .describe("Number of memories before the anchor"),
            depth_after: z
                .number()
                .int()
                .min(0)
                .max(50)
                .default(7)
                .describe("Number of memories after the anchor"),
        },
        async (args) => {
            const anchor = await q.get_mem.get(args.memory_id);
            if (!anchor) {
                return {
                    content: [
                        {
                            type: "text",
                            text: `Memory ${args.memory_id} not found`,
                        },
                    ],
                    isError: true,
                };
            }

            const project = (anchor as any).project || "default";
            const created = Number((anchor as any).created_at || 0);

            const before = await q.timeline_before.all(
                project,
                created,
                args.depth_before,
            );
            const after = await q.timeline_after.all(
                project,
                created,
                args.depth_after,
            );

            const timeline = [...before.reverse(), anchor, ...after];
            return {
                content: [
                    {
                        type: "text",
                        text: JSON.stringify({ timeline }, null, 2),
                    },
                ],
            };
        },
    );

    registry.tool(
        "openmemory_get",
        "Fetch a single memory by identifier",
        {
            id: z.string().min(1).describe("Memory identifier to load"),
            include_vectors: z
                .boolean()
                .default(false)
                .describe("Include sector vector metadata"),
            user_id: z
                .string()
                .trim()
                .min(1)
                .optional()
                .describe(
                    "Validate ownership against a specific user identifier",
                ),
        },
        async ({ id, include_vectors, user_id }) => {
            const u = uid(user_id);
            const mem = await q.get_mem.get(id);
            if (!mem)
                return {
                    content: [
                        { type: "text", text: `Memory ${id} not found.` },
                    ],
                };
            if (u && mem.user_id !== u)
                return {
                    content: [
                        {
                            type: "text",
                            text: `Memory ${id} not found for user ${u}.`,
                        },
                    ],
                };
            const vecs = include_vectors
                ? await vector_store.getVectorsById(id)
                : [];
            const pay = {
                id: mem.id,
                content: mem.content,
                primary_sector: mem.primary_sector,
                salience: mem.salience,
                decay_lambda: mem.decay_lambda,
                created_at: mem.created_at,
                updated_at: mem.updated_at,
                last_seen_at: mem.last_seen_at,
                user_id: mem.user_id,
                tags: p(mem.tags || "[]"),
                metadata: p(mem.meta || "{}"),
                sectors: include_vectors
                    ? vecs.map((v) => v.sector)
                    : undefined,
            };
            return {
                content: [{ type: "text", text: JSON.stringify(pay, null, 2) }],
            };
        },
    );
    // ── openmemory_status (self-teaching: embeds protocol + stats in response) ──
    registry.tool(
        "openmemory_consolidate",
        "Check if a project's memories need consolidation and return the lowest-value candidates for merging. Call this periodically to prevent memory bloat. The agent should review the candidates, synthesize them into fewer high-quality memories via openmemory_store, then delete the originals via openmemory_delete.",
        {
            project: z.string().trim().min(1).describe("Project to check for consolidation"),
            threshold: z
                .number()
                .int()
                .min(10)
                .max(500)
                .default(50)
                .describe("Consolidation triggers when memory count exceeds this (default 50)"),
            candidate_count: z
                .number()
                .int()
                .min(5)
                .max(30)
                .default(10)
                .describe("Number of lowest-salience candidates to return (default 10)"),
        },
        async (args) => {
            const count_row = await q.count_by_project.get(args.project);
            const total = count_row?.c ?? 0;

            if (total <= (args.threshold ?? 50)) {
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify({
                            needs_consolidation: false,
                            total_memories: total,
                            threshold: args.threshold ?? 50,
                            message: `Project "${args.project}" has ${total} memories (threshold: ${args.threshold ?? 50}). No consolidation needed.`,
                        }, null, 2),
                    }],
                };
            }

            const candidates = await q.get_oldest_by_project.all(
                args.project,
                args.candidate_count ?? 10,
            );

            const formatted = candidates.map((c: any) => ({
                id: c.id,
                content: c.content,
                observation_type: c.observation_type,
                primary_sector: c.primary_sector,
                salience: c.salience,
                tags: p(c.tags || "[]"),
                metadata: p(c.meta || "{}"),
                created_at: c.created_at,
            }));

            return {
                content: [{
                    type: "text",
                    text: JSON.stringify({
                        needs_consolidation: true,
                        total_memories: total,
                        threshold: args.threshold ?? 50,
                        excess: total - (args.threshold ?? 50),
                        candidates: formatted,
                        instructions: "Review these low-salience memories. Synthesize related ones into fewer, higher-quality memories using openmemory_store, then delete the originals using openmemory_delete. Preserve important facts and decisions; discard redundant or outdated observations.",
                    }, null, 2),
                }],
            };
        },
    );

    registry.tool(
        "openmemory_status",
        "Get memory system status, statistics, and usage protocol. Call this on wake-up to learn how to use the memory system.",
        {},
        async () => {
            const total_rows = await all_async(
                `select count(*) as c from ${memories_table}`,
            );
            const total = total_rows?.[0]?.c ?? 0;

            const by_sector = await all_async(
                `select primary_sector as sector, count(*) as count, round(avg(salience), 3) as avg_salience from ${memories_table} group by primary_sector order by count desc`,
            );

            const by_project = await all_async(
                `select project, count(*) as count from ${memories_table} group by project order by count desc limit 20`,
            );

            const by_type = await all_async(
                `select observation_type as type, count(*) as count from ${memories_table} group by observation_type order by count desc`,
            );

            const summary_count = await all_async(
                `select count(*) as c from summaries`,
            );

            const session_count = await all_async(
                `select count(*) as c from sessions`,
            );

            const protocol = `MEMORY PROTOCOL:
1. SESSION START: Call openmemory_wakeup(project) to load compressed context (~200 tokens). If no memories exist, proceed normally.
2. DURING WORK: After completing meaningful tasks, call openmemory_store with:
   - content: compressed observation (write for a future agent with zero context)
   - project: working directory basename
   - observation_type: one of observation|bugfix|decision|discovery|feature|gotcha|refactor
   - tags: consistent category tags
   - metadata: { facts: [...], files_involved: [...], concepts: [...] }
3. BEFORE FACTS CHANGE: If a previously stored fact is now outdated, store the new fact with updated context.
4. SESSION END: Call openmemory_summarize with session_id, project, request, completed, learned, next_steps, files_modified.
5. SEARCH: Use openmemory_query(query, project) when user asks about past work. Use natural language.
6. NEVER GUESS: If the user asks about past work, search first. Don't rely on your own memory.
7. CONSOLIDATION: When openmemory_wakeup shows many memories, call openmemory_consolidate(project) to check if consolidation is needed. If yes, synthesize low-value memories into fewer high-quality ones, then delete originals.`;

            const pay = {
                total_memories: total,
                total_summaries: summary_count?.[0]?.c ?? 0,
                total_sessions: session_count?.[0]?.c ?? 0,
                by_sector,
                by_project,
                by_observation_type: by_type,
                embeddings: getEmbeddingInfo(),
                protocol,
                available_tools: [
                    "openmemory_status",
                    "openmemory_wakeup",
                    "openmemory_query",
                    "openmemory_store",
                    "openmemory_consolidate",
                    "openmemory_summarize",
                    "openmemory_timeline",
                    "openmemory_list",
                    "openmemory_get",
                    "openmemory_reinforce",
                    "openmemory_delete",
                ],
            };

            return {
                content: [{ type: "text", text: JSON.stringify(pay, null, 2) }],
            };
        },
    );

    // ── openmemory_wakeup (4-layer: L0 identity + L1 top-N essential story) ──
    registry.tool(
        "openmemory_wakeup",
        "Load compressed project context for session start. Returns identity + top memories + recent summaries in a single call (~200 tokens). Call this ONCE at the beginning of every session.",
        {
            project: z
                .string()
                .trim()
                .min(1)
                .describe("Project name (working directory basename). Use 'home' for home directory."),
            limit: z
                .number()
                .int()
                .min(1)
                .max(50)
                .default(15)
                .describe("Max memories to include in essential story (default 15)"),
        },
        async (args: { project: string; limit?: number }) => {
            const limit = args.limit ?? 15;
            const max_chars = 3200;

            // L0: Identity (static, optional)
            let identity = "";
            try {
                const fs = await import("fs");
                const path = await import("path");
                const os = await import("os");
                const id_path = path.join(os.homedir(), ".openmemory-enhanced", "identity.txt");
                if (fs.existsSync(id_path)) {
                    identity = fs.readFileSync(id_path, "utf-8").trim();
                }
            } catch { /* no identity file, that's fine */ }

            // L1: Top-N memories by salience, project-scoped
            const top_memories = await all_async(
                `select id, content, observation_type, primary_sector, salience, project from ${memories_table} where project=? order by salience desc, created_at desc limit ?`,
                [args.project, limit],
            );

            // Group by observation_type, truncate to budget
            const grouped: Record<string, string[]> = {};
            let chars = 0;
            for (const m of top_memories) {
                if (chars >= max_chars) break;
                const typ = m.observation_type || "observation";
                if (!grouped[typ]) grouped[typ] = [];
                const line = trunc(m.content.replace(/\s+/g, " ").trim(), 200);
                grouped[typ].push(line);
                chars += line.length;
            }

            let essential_story = "";
            for (const [typ, lines] of Object.entries(grouped)) {
                essential_story += `[${typ.toUpperCase()}]\n`;
                for (const line of lines) {
                    essential_story += `- ${line}\n`;
                }
                essential_story += "\n";
            }

            // L1.5: Recent summaries
            const recent_summaries = await all_async(
                `select request, completed, learned, next_steps, created_at from summaries where project=? order by created_at desc limit 3`,
                [args.project],
            );

            let summary_text = "";
            for (const s of recent_summaries) {
                summary_text += `Session: ${s.request || "?"}\n`;
                summary_text += `  Done: ${trunc(s.completed || "", 150)}\n`;
                if (s.learned) summary_text += `  Learned: ${trunc(s.learned, 100)}\n`;
                if (s.next_steps) summary_text += `  Next: ${trunc(s.next_steps, 100)}\n`;
                summary_text += "\n";
            }

            const sections: string[] = [];
            if (identity) sections.push(`## Identity\n${identity}`);
            if (essential_story.trim()) sections.push(`## Essential Context (${top_memories.length} memories)\n${essential_story.trim()}`);
            if (summary_text.trim()) sections.push(`## Recent Sessions (${recent_summaries.length})\n${summary_text.trim()}`);

            if (sections.length === 0) {
                return {
                    content: [{ type: "text", text: `No memories found for project "${args.project}". This appears to be a new project.` }],
                };
            }

            return {
                content: [{ type: "text", text: sections.join("\n\n") }],
            };
        },
    );

    registry.apply(srv);

    srv.resource(
        "openmemory-config",
        "openmemory://config",
        {
            mimeType: "application/json",
            description:
                "Runtime configuration snapshot for the OpenMemory MCP server",
        },
        async () => {
            const stats = await all_async(
                `select primary_sector as sector, count(*) as count, avg(salience) as avg_salience from ${memories_table} group by primary_sector`,
            );
            const pay = {
                mode: env.mode,
                sectors: sector_configs,
                stats,
                embeddings: getEmbeddingInfo(),
                server: { version: "2.1.0", protocol: "2025-06-18" },
                available_tools: [
                    "openmemory_status",
                    "openmemory_wakeup",
                    "openmemory_query",
                    "openmemory_store",
                    "openmemory_reinforce",
                    "openmemory_list",
                    "openmemory_get",
                    "openmemory_delete",
                    "openmemory_summarize",
                    "openmemory_timeline",
                ],
            };
            return {
                contents: [
                    {
                        uri: "openmemory://config",
                        text: JSON.stringify(pay, null, 2),
                    },
                ],
            };
        },
    );

    srv.server.oninitialized = () => {

        console.error(
            "[MCP] initialization completed with client:",
            srv.server.getClientVersion(),
        );
    };
    return srv;
};

const extract_pay = async (req: IncomingMessage & { body?: any }) => {
    if (req.body !== undefined) {
        if (typeof req.body === "string") {
            if (!req.body.trim()) return undefined;
            return JSON.parse(req.body);
        }
        if (typeof req.body === "object" && req.body !== null) return req.body;
        return undefined;
    }
    const raw = await new Promise<string>((resolve, reject) => {
        let buf = "";
        req.on("data", (chunk) => {
            buf += chunk;
        });
        req.on("end", () => resolve(buf));
        req.on("error", reject);
    });
    if (!raw.trim()) return undefined;
    return JSON.parse(raw);
};

export const mcp = (app: any) => {
    const handle_req = async (req: any, res: any) => {
        try {
            const pay = await extract_pay(req);
            if (!pay || typeof pay !== "object") {
                send_err(res, -32600, "Request body must be a JSON object");
                return;
            }
            console.error("[MCP] Incoming request:", JSON.stringify(pay));
            set_hdrs(res);

            // Create a fresh transport + server per request to support
            // multiple clients (MCP SDK 1.27 rejects re-initialization
            // on a single transport instance).
            const srv = create_mcp_srv();
            const trans = new StreamableHTTPServerTransport({
                sessionIdGenerator: undefined,
                enableJsonResponse: true,
            });
            await srv.connect(trans);
            await trans.handleRequest(req, res, pay);
        } catch (error) {
            console.error("[MCP] Error handling request:", error);
            if (error instanceof SyntaxError) {
                send_err(res, -32600, "Invalid JSON payload");
                return;
            }
            if (!res.headersSent)
                send_err(
                    res,
                    -32603,
                    "Internal server error",
                    (error as any)?.id ?? null,
                    500,
                );
        }
    };

    app.post("/mcp", (req: any, res: any) => {
        void handle_req(req, res);
    });
    app.options("/mcp", (_req: any, res: any) => {
        res.statusCode = 204;
        set_hdrs(res);
        res.end();
    });

    const method_not_allowed = (_req: IncomingMessage, res: ServerResponse) => {
        send_err(
            res,
            -32600,
            "Method not supported. Use POST  /mcp with JSON payload.",
            null,
            405,
        );
    };
    app.get("/mcp", method_not_allowed);
    app.delete("/mcp", method_not_allowed);
    app.put("/mcp", method_not_allowed);
};

export const start_mcp_stdio = async () => {
    const srv = create_mcp_srv();
    const trans = new StdioServerTransport();
    await srv.connect(trans);

};

if (typeof require !== "undefined" && require.main === module) {
    void start_mcp_stdio().catch((error) => {
        console.error("[MCP] STDIO startup failed:", error);
        process.exitCode = 1;
    });
}
