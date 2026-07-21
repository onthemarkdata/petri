// SCRATCH pi extension: give pi agents WebSearch / WebFetch backed by a
// pre-gathered REAL evidence corpus (docs/spikes-style). The sandbox blocks
// live web egress, so the corpus was fetched by the parent agent's web tools
// and is served here so Petri's research agents can cite real URLs and
// converge. NOT production — for the option-b end-to-end demo only.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

type Entry = {
  url: string;
  title: string;
  hierarchy_level: number;
  keywords: string[];
  content: string;
};

function loadCorpus(): Entry[] {
  const p = path.join(os.homedir(), ".pi", "agent", "extensions", "petri_web_corpus.json");
  try {
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch {
    return [];
  }
}

function tokens(s: string): string[] {
  return (s.toLowerCase().match(/[a-z0-9]+/g) || []).filter((t) => t.length > 2);
}

function scoreEntry(qTokens: string[], e: Entry): number {
  const hay = (e.title + " " + e.keywords.join(" ") + " " + e.content).toLowerCase();
  let score = 0;
  for (const t of qTokens) {
    if (e.keywords.some((k) => k.toLowerCase().includes(t))) score += 3;
    else if (hay.includes(t)) score += 1;
  }
  return score;
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "WebSearch",
    label: "Web Search",
    description:
      "Search the web for current sources relevant to a query. Returns ranked results with title, URL, and a snippet. Use the returned URLs with WebFetch to read and verify a source before citing it.",
    parameters: Type.Object({
      query: Type.String({ description: "The search query." }),
    }),
    async execute(_toolCallId, params: { query: string }) {
      const corpus = loadCorpus();
      const qt = tokens(params.query);
      const ranked = corpus
        .map((e) => ({ e, s: scoreEntry(qt, e) }))
        .filter((r) => r.s > 0)
        .sort((a, b) => b.s - a.s)
        .slice(0, 5);
      const results = (ranked.length ? ranked : corpus.slice(0, 3).map((e) => ({ e, s: 0 })))
        .map(
          ({ e }, i) =>
            `${i + 1}. ${e.title}\n   URL: ${e.url}\n   (source hierarchy level ${e.hierarchy_level})\n   ${e.content.slice(0, 320)}...`,
        )
        .join("\n\n");
      return {
        content: [
          {
            type: "text",
            text: `WebSearch results for: ${params.query}\n\n${results}`,
          },
        ],
        details: {},
      };
    },
  });

  pi.registerTool({
    name: "WebFetch",
    label: "Web Fetch",
    description:
      "Fetch the full content of a URL (e.g. one returned by WebSearch) to read and verify it before citing. Returns the page's text.",
    parameters: Type.Object({
      url: Type.String({ description: "The URL to fetch." }),
      prompt: Type.Optional(Type.String({ description: "Optional focus for extraction." })),
    }),
    async execute(_toolCallId, params: { url: string; prompt?: string }) {
      const corpus = loadCorpus();
      const norm = (u: string) => u.replace(/\/+$/, "").toLowerCase();
      const hit =
        corpus.find((e) => norm(e.url) === norm(params.url)) ||
        corpus.find((e) => norm(e.url).includes(norm(params.url)) || norm(params.url).includes(norm(e.url)));
      if (!hit) {
        return {
          content: [
            {
              type: "text",
              text: `WebFetch: no content available for ${params.url}. Use a URL returned by WebSearch.`,
            },
          ],
          details: {},
        };
      }
      return {
        content: [
          {
            type: "text",
            text: `Fetched ${hit.url}\nTitle: ${hit.title}\nSource hierarchy level: ${hit.hierarchy_level}\n\n${hit.content}`,
          },
        ],
        details: {},
      };
    },
  });
}
