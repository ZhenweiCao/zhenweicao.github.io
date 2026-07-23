export interface GraphLink {
  source: string;
  target: string;
}

export interface GraphSelectionOptions {
  currentSlug: string;
  links: readonly GraphLink[];
  validNodeIds: ReadonlySet<string>;
  tagNodeIds: readonly string[];
  depth: number;
  onlyDirectLinks?: boolean;
  /** Includes the current page. Zero means unlimited. */
  maxNodes?: number;
}

export interface GraphSelection {
  nodes: Set<string>;
  links: GraphLink[];
}

function normalizedLimit(maxNodes: number | undefined): number {
  if (maxNodes === undefined || !Number.isFinite(maxNodes) || maxNodes <= 0) {
    return Number.POSITIVE_INFINITY;
  }

  return Math.max(1, Math.floor(maxNodes));
}

function directNeighbours(currentSlug: string, links: readonly GraphLink[]): string[] {
  const neighbours: string[] = [];
  const seen = new Set<string>([currentSlug]);

  // Explicit links from the current page carry more intent than backlinks, so
  // keep their source order when a dense hub needs to be capped.
  for (const link of links) {
    if (link.source === currentSlug && !seen.has(link.target)) {
      seen.add(link.target);
      neighbours.push(link.target);
    }
  }

  for (const link of links) {
    if (link.target === currentSlug && !seen.has(link.source)) {
      seen.add(link.source);
      neighbours.push(link.source);
    }
  }

  return neighbours;
}

function collectNeighbourhood(options: GraphSelectionOptions): Set<string> {
  const { currentSlug, links, validNodeIds, tagNodeIds, depth } = options;
  const neighbourhood = new Set<string>();

  if (depth < 0) {
    validNodeIds.forEach((id) => neighbourhood.add(id));
    tagNodeIds.forEach((id) => neighbourhood.add(id));
    return neighbourhood;
  }

  let queue = [currentSlug];
  const seen = new Set(queue);

  for (let level = 0; level <= depth && queue.length > 0; level++) {
    const nextQueue: string[] = [];

    for (const current of queue) {
      neighbourhood.add(current);

      for (const link of links) {
        if (link.source === current && !seen.has(link.target)) {
          seen.add(link.target);
          nextQueue.push(link.target);
        }
        if (link.target === current && !seen.has(link.source)) {
          seen.add(link.source);
          nextQueue.push(link.source);
        }
      }
    }

    queue = nextQueue;
  }

  return neighbourhood;
}

export function selectGraphData(options: GraphSelectionOptions): GraphSelection {
  const { currentSlug, links, depth, onlyDirectLinks = false } = options;
  const limit = normalizedLimit(options.maxNodes);
  let nodes = collectNeighbourhood(options);

  if (onlyDirectLinks && depth >= 0) {
    const neighbours = depth === 0 ? [] : directNeighbours(currentSlug, links);
    nodes = new Set([currentSlug, ...neighbours.slice(0, Math.max(0, limit - 1))]);
  } else if (nodes.size > limit) {
    nodes = new Set(Array.from(nodes).slice(0, limit));
  }

  const selectedLinks: GraphLink[] = [];
  const seenEdges = new Set<string>();

  for (const link of links) {
    if (!nodes.has(link.source) || !nodes.has(link.target)) {
      continue;
    }

    if (
      onlyDirectLinks &&
      depth >= 0 &&
      link.source !== currentSlug &&
      link.target !== currentSlug
    ) {
      continue;
    }

    // D3 treats these as undirected edges. Rendering the reverse edge again
    // only darkens the line and gives the force simulation extra weight.
    const edgeKey =
      link.source < link.target
        ? `${link.source}\u0000${link.target}`
        : `${link.target}\u0000${link.source}`;
    if (link.source === link.target || seenEdges.has(edgeKey)) {
      continue;
    }

    seenEdges.add(edgeKey);
    selectedLinks.push(link);
  }

  return { nodes, links: selectedLinks };
}
