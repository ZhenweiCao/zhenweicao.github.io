export interface RenderEpoch {
  begin(): number;
  invalidate(): void;
  isCurrent(epoch: number): boolean;
}

export function createRenderEpoch(): RenderEpoch {
  let currentEpoch = 0;

  return {
    begin() {
      currentEpoch += 1;
      return currentEpoch;
    },
    invalidate() {
      currentEpoch += 1;
    },
    isCurrent(epoch) {
      return epoch === currentEpoch;
    },
  };
}
