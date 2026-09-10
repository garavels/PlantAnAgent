import { useCallback, useEffect, useRef, useState } from "react";
import { useV2Demo } from "../hooks/useV2Demo";
import type {
  PublicRecording,
  RecordingSummary,
  V2DemoFrame,
} from "../types/v2";
import type { UnitId } from "../components/plant-3d/selection";
import { checkRecording, readCatalog, readFrame, readRecording } from "./api";

export function useWorkspaceState() {
  const live = useV2Demo();
  const [recording, setRecording] = useState<PublicRecording | null>(null);
  const [catalog, setCatalog] = useState<RecordingSummary[]>([]);
  const [historical, setHistorical] = useState<V2DemoFrame | null>(null);
  const [selectedUnit, setSelectedUnit] = useState<UnitId | null>(
    "electrolyser",
  );
  const [playing, setPlaying] = useState(false);
  const [running, setRunning] = useState(false);
  const [reading, setReading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const generation = useRef(0);
  const liveRun = useRef<string | undefined>(undefined);
  const cache = useRef(new Map<string, V2DemoFrame>());
  const [defaultLoaded, setDefaultLoaded] = useState(false);
  const snapshot = recording?.snapshot ?? live.snapshot;
  const frame = historical ?? snapshot?.frame ?? null;
  const index = frame?.clock.decision_index ?? 0;
  const latestIndex = snapshot?.frame.clock.decision_index ?? 0;
  const busy = live.pending !== null || reading;

  useEffect(() => {
    const id = live.snapshot?.run_id;
    if (id && liveRun.current && id !== liveRun.current && !recording) {
      ++generation.current;
      setHistorical(null);
      setReading(false);
      setRunning(false);
      cache.current.clear();
    }
    liveRun.current = id;
  }, [live.snapshot?.run_id, recording]);
  useEffect(
    () => () => {
      ++generation.current;
    },
    [],
  );

  const reloadCatalog = useCallback(async () => {
    try {
      const items = await readCatalog();
      setCatalog(items);
      return items;
    } catch {
      return [] as RecordingSummary[];
    }
  }, []);

  const loadRecording = useCallback(async (id?: string, start = 0) => {
    const request = ++generation.current;
    setRunning(false);
    setPlaying(false);
    setReading(true);
    setNotice(null);
    try {
      const result = checkRecording(await readRecording(id));
      if (request !== generation.current) return;
      setRecording(result);
      setHistorical(result.frames[Math.min(start, result.frames.length - 1)]);
    } catch (e) {
      if (request === generation.current) setNotice(String(e));
    } finally {
      if (request === generation.current) setReading(false);
    }
  }, []);

  useEffect(() => {
    void reloadCatalog();
  }, [reloadCatalog]);
  useEffect(() => {
    if (defaultLoaded || !catalog.length) return;
    setDefaultLoaded(true);
    const featured =
      catalog.find(
        (c) => c.available && c.source === "saved" && c.controller === "E-beta",
      ) ?? catalog.find((c) => c.available && c.source === "saved");
    if (featured) void loadRecording(featured.id, 32);
  }, [catalog, defaultLoaded, loadRecording]);

  const selectFrame = useCallback(
    async (target: number) => {
      if (
        !snapshot ||
        target < 0 ||
        target > snapshot.frame.clock.decision_index
      )
        return;
      setRunning(false);
      const request = ++generation.current;
      if (recording) {
        setHistorical(recording.frames[target]);
        setReading(false);
        return;
      }
      if (target === snapshot.frame.clock.decision_index) {
        setHistorical(null);
        setReading(false);
        return;
      }
      const key = `${snapshot.run_id}:${target}`;
      const cached = cache.current.get(key);
      if (cached) {
        setHistorical(cached);
        setReading(false);
        return;
      }
      setReading(true);
      setNotice(null);
      try {
        const value = await readFrame(snapshot.run_id, target);
        if (request !== generation.current) return;
        cache.current.set(key, value);
        setHistorical(value);
      } catch (e) {
        if (request === generation.current) {
          setNotice(String(e));
          setPlaying(false);
        }
      } finally {
        if (request === generation.current) setReading(false);
      }
    },
    [snapshot, recording],
  );

  useEffect(() => {
    if (!playing || !recording) return;
    if (index >= latestIndex) {
      setPlaying(false);
      return;
    }
    const timer = setTimeout(() => void selectFrame(index + 1), 600);
    return () => clearTimeout(timer);
  }, [playing, recording, index, latestIndex, selectFrame]);

  useEffect(() => {
    if (!running || recording || live.pending || reading) return;
    if (live.error || live.stale || live.snapshot?.status !== "ready") {
      setRunning(false);
      return;
    }
    const timer = setTimeout(() => void live.advance(1), 350);
    return () => clearTimeout(timer);
  }, [running, recording, reading, live]);

  const useSimulation = () => {
    ++generation.current;
    setDefaultLoaded(true);
    setRecording(null);
    setHistorical(null);
    setReading(false);
    setPlaying(false);
    setRunning(false);
    setNotice(null);
    void live.refresh();
  };
  const reset = async (controller: "H" | "E-beta") => {
    ++generation.current;
    setDefaultLoaded(true);
    setPlaying(false);
    setRunning(false);
    setRecording(null);
    setHistorical(null);
    setReading(false);
    cache.current.clear();
    await live.reset(controller);
    await reloadCatalog();
  };
  const advance = (count: number) => {
    setHistorical(null);
    setPlaying(false);
    return live.advance(count);
  };
  const exportRecording = async () => {
    try {
      const result = recording ?? (await readRecording());
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(result)], { type: "application/json" }),
      );
      const a = document.createElement("a");
      a.href = url;
      a.download = `plant-${result.snapshot.controller.id}-${result.snapshot.run_id}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setNotice(String(e));
    }
  };
  return {
    live,
    snapshot,
    frame,
    index,
    latestIndex,
    recording,
    catalog,
    historical,
    selectedUnit,
    setSelectedUnit,
    playing,
    setPlaying,
    running,
    setRunning,
    busy,
    reading,
    notice,
    reloadCatalog,
    loadRecording,
    selectFrame,
    useSimulation,
    reset,
    advance,
    exportRecording,
  };
}
