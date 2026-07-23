export const graphPanelStorageKey = "graph-panel-collapsed";

export interface GraphPanelStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export function graphPanelIsCollapsed(storage?: GraphPanelStorage | null): boolean {
  if (!storage) return false;

  try {
    return storage.getItem(graphPanelStorageKey) === "true";
  } catch {
    return false;
  }
}

export function storeGraphPanelState(
  storage: GraphPanelStorage | null | undefined,
  collapsed: boolean,
): void {
  try {
    storage?.setItem(graphPanelStorageKey, String(collapsed));
  } catch {
    // Storage can be unavailable in private browsing or restricted embeds.
  }
}
