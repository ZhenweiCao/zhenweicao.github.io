import { describe, expect, it } from "vitest";
import Graph from "../src/components/Graph";
import { selectGraphData } from "../src/components/graph-data";
import {
  graphPanelIsCollapsed,
  graphPanelStorageKey,
  storeGraphPanelState,
} from "../src/components/graph-panel";
import { createRenderEpoch } from "../src/components/graph-lifecycle";

describe("focused graph behavior", () => {
  it("should create a Graph component with default options", () => {
    const component = Graph({});

    expect(component).toBeDefined();
    expect(typeof component).toBe("function");
  });

  it("should create a Graph component with custom options", () => {
    const component = Graph({
      localGraph: {
        depth: 2,
        drag: false,
        zoom: true,
      },
      globalGraph: {
        depth: -1,
        focusOnHover: true,
      },
    });

    expect(component).toBeDefined();
    expect(typeof component).toBe("function");
  });

  it("should export component with css property", () => {
    const component = Graph({});

    expect(component.css).toBeDefined();
    expect(typeof component.css).toBe("string");
  });

  it("should export component with afterDOMLoaded script", () => {
    const component = Graph({});

    expect(component.afterDOMLoaded).toBeDefined();
    expect(typeof component.afterDOMLoaded).toBe("string");
  });

  it("persists the collapsed panel preference", () => {
    const values = new Map<string, string>();
    const storage = {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
    };

    expect(graphPanelIsCollapsed(storage)).toBe(false);
    storeGraphPanelState(storage, true);
    expect(values.get(graphPanelStorageKey)).toBe("true");
    expect(graphPanelIsCollapsed(storage)).toBe(true);
  });

  it("falls back safely when browser storage is unavailable", () => {
    const storage = {
      getItem: () => {
        throw new Error("blocked");
      },
      setItem: () => {
        throw new Error("blocked");
      },
    };

    expect(graphPanelIsCollapsed(storage)).toBe(false);
    expect(() => storeGraphPanelState(storage, true)).not.toThrow();
  });

  it("invalidates an in-flight render epoch", () => {
    const lifecycle = createRenderEpoch();
    const firstRender = lifecycle.begin();

    expect(lifecycle.isCurrent(firstRender)).toBe(true);
    lifecycle.invalidate();
    expect(lifecycle.isCurrent(firstRender)).toBe(false);
  });

  it("keeps only current-page edges in a focused local graph", () => {
    const selection = selectGraphData({
      currentSlug: "index",
      links: [
        { source: "index", target: "a" },
        { source: "index", target: "b" },
        { source: "a", target: "b" },
      ],
      validNodeIds: new Set(["index", "a", "b"]),
      tagNodeIds: [],
      depth: 1,
      onlyDirectLinks: true,
    });

    expect(selection.nodes).toEqual(new Set(["index", "a", "b"]));
    expect(selection.links).toEqual([
      { source: "index", target: "a" },
      { source: "index", target: "b" },
    ]);
  });

  it("prioritizes outgoing links before backlinks when capped", () => {
    const selection = selectGraphData({
      currentSlug: "index",
      links: [
        { source: "incoming", target: "index" },
        { source: "index", target: "first" },
        { source: "index", target: "second" },
      ],
      validNodeIds: new Set(["index", "incoming", "first", "second"]),
      tagNodeIds: [],
      depth: 1,
      onlyDirectLinks: true,
      maxNodes: 3,
    });

    expect(selection.nodes).toEqual(new Set(["index", "first", "second"]));
  });

  it("collapses reverse links into one visual edge", () => {
    const selection = selectGraphData({
      currentSlug: "index",
      links: [
        { source: "index", target: "a" },
        { source: "a", target: "index" },
      ],
      validNodeIds: new Set(["index", "a"]),
      tagNodeIds: [],
      depth: 1,
      onlyDirectLinks: true,
    });

    expect(selection.links).toEqual([{ source: "index", target: "a" }]);
  });

  it("keeps every page and tag in an unlimited global graph", () => {
    const selection = selectGraphData({
      currentSlug: "index",
      links: [{ source: "a", target: "tags/topic" }],
      validNodeIds: new Set(["index", "a"]),
      tagNodeIds: ["tags/topic"],
      depth: -1,
      maxNodes: 0,
    });

    expect(selection.nodes).toEqual(new Set(["index", "a", "tags/topic"]));
    expect(selection.links).toEqual([{ source: "a", target: "tags/topic" }]);
  });
});
