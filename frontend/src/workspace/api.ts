import type {
  PublicRecording,
  RecordingSummary,
  V2DemoFrame,
} from "../types/v2";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`/api/v2/demo${path}`, { cache: "no-store" });
  if (!res.ok)
    throw new Error(
      `Unable to read the recording (${res.status}). Refresh the current run.`,
    );
  return res.json() as Promise<T>;
}
export const readFrame = (run: string, index: number) =>
  get<V2DemoFrame>(`/frames/${encodeURIComponent(run)}/${index}`);
export const readCatalog = () => get<RecordingSummary[]>("/recordings");
export const readRecording = (id?: string) =>
  get<PublicRecording>(
    id ? `/recordings/${encodeURIComponent(id)}` : "/recording",
  );

export function checkRecording(record: PublicRecording) {
  if (
    record.schema !== "plant-public-recording-v1" ||
    !record.snapshot ||
    !Array.isArray(record.frames) ||
    !record.frames.length ||
    record.frames.length > 673 ||
    record.snapshot.environment.fixture_seed !== 9_000_034 ||
    record.frames.some(
      (f, i) => f.clock.decision_index !== i || f.clock.elapsed_hours !== i / 4,
    )
  ) {
    throw new Error("This is not a supported public fixture recording.");
  }
  return record;
}
