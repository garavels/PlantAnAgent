import type {
  PublicRecording,
  RecordingSummary,
  V2DemoFrame,
} from "../types/v2";
export { checkRecording } from "./recording";

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
