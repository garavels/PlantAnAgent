import { useCallback, useEffect, useRef, useState } from "react";
import { useV2Demo } from "../hooks/useV2Demo";
import type {
  PublicRecording,
  RecordingSummary,
  V2DemoFrame,
} from "../types/v2";
import type { UnitId } from "../components/plant-3d/selection";
import { checkRecording, readCatalog, readFrame, readRecording } from "./api";
import { featuredRecording } from "./recording";
import { siteTime, time as fixtureTime } from "./format";
import type { SiteActivationRequest } from "./siteTypes";
import { useAgentWalkthrough } from "./useAgentWalkthrough";

export function useWorkspaceState() {
  const live = useV2Demo();
  const [recording, setRecording] = useState<PublicRecording | null>(null);
  const [catalog, setCatalog] = useState<RecordingSummary[]>([]);
  const [catalogStatus, setCatalogStatus] = useState<"loading" | "ready" | "error">("loading");
  const [historical, setHistorical] = useState<V2DemoFrame | null>(null);
  const [selectedUnit, setSelectedUnit] = useState<UnitId | null>(
    "electrolyser",
  );
  const [playing, setPlaying] = useState(false);
  const [running, setRunning] = useState(false);
  const [reading, setReading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const generation = useRef(0);
  const catalogGeneration = useRef(0);
  const liveRun = useRef<string | undefined>(undefined);
  const cache = useRef(new Map<string, V2DemoFrame>());
  const defaultLoaded = useRef(false);
  const snapshot = recording?.snapshot ?? live.snapshot;
  const frame = historical ?? snapshot?.frame ?? null;
  const index = frame?.clock.decision_index ?? 0;
  const latestIndex = snapshot?.frame.clock.decision_index ?? 0;
  const busy = (!recording && live.pending !== null) || reading;
  const agent = useAgentWalkthrough(frame, snapshot?.run_id, (playing || running) && !busy);
  const siteForecast = snapshot?.mode === "site_forecast" ? snapshot.site_forecast : undefined;
  const time = (hours: number) => siteForecast ? siteTime(siteForecast.start_time, hours) : fixtureTime(hours);

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
      ++catalogGeneration.current;
    },
    [],
  );

  const reloadCatalog = useCallback(async () => {
    const request = ++catalogGeneration.current;
    setCatalogStatus("loading");
    try {
      const items = await readCatalog();
      if (request !== catalogGeneration.current) return [] as RecordingSummary[];
      setCatalog(items);
      setCatalogStatus("ready");
      return items;
    } catch {
      if (request === catalogGeneration.current) setCatalogStatus("error");
      return [] as RecordingSummary[];
    }
  }, []);

  const loadRecording = useCallback(async (id?: string, start = 0): Promise<boolean> => {
    // Claim selection synchronously so a delayed default effect cannot replace it.
    defaultLoaded.current = true;
    const request = ++generation.current;
    setRunning(false);
    setPlaying(false);
    setReading(true);
    setNotice(null);
    try {
      const result = checkRecording(await readRecording(id));
      if (request !== generation.current) return false;
      setRecording(result);
      setHistorical(result.frames[Math.max(0, Math.min(Math.trunc(start), result.frames.length - 1))]);
      return true;
    } catch (e) {
      if (request === generation.current) setNotice(String(e));
      return false;
    } finally {
      if (request === generation.current) setReading(false);
    }
  }, []);

  useEffect(() => {
    void reloadCatalog();
  }, [reloadCatalog]);
  useEffect(() => {
    if (defaultLoaded.current || !live.snapshot) return;
    if (live.snapshot.mode === "site_forecast") {
      defaultLoaded.current = true;
      return;
    }
    if (!catalog.length) return;
    defaultLoaded.current = true;
    const featured =
      featuredRecording(catalog) ?? catalog.find(
        (c) => c.available && c.source === "saved" && c.controller === "MPC",
      ) ?? catalog.find((c) => c.available && c.source === "saved");
    if (featured) void loadRecording(featured.id, featured.mode === "recorded_research" ? 0 : 32);
  }, [catalog, loadRecording, live.snapshot]);

  const selectFrame = useCallback(
    async (target: number) => {
      if (
        !snapshot || !Number.isInteger(target) ||
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
    if (!playing || !recording || !agent.ready) return;
    if (index >= latestIndex) {
      setPlaying(false);
      return;
    }
    const timer = setTimeout(() => void selectFrame(index + 1), 600);
    return () => clearTimeout(timer);
  }, [playing, recording, index, latestIndex, selectFrame, agent.ready]);

  useEffect(() => {
    if (!running || recording || live.pending || reading || !agent.ready) return;
    if (live.error || live.stale || live.snapshot?.status !== "ready") {
      setRunning(false);
      return;
    }
    const timer = setTimeout(() => void live.advance(1), 350);
    return () => clearTimeout(timer);
  }, [running, recording, reading, live, agent.ready]);

  const useSimulation = () => {
    ++generation.current;
    defaultLoaded.current = true;
    setRecording(null);
    setHistorical(null);
    setReading(false);
    setPlaying(false);
    setRunning(false);
    setNotice(null);
    void live.refresh();
  };
  const reset = async (controller: "H" | "MPC") => {
    ++generation.current;
    defaultLoaded.current = true;
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
    if (recording) return selectFrame(Math.min(latestIndex, index + count));
    setHistorical(null);
    setPlaying(false);
    return live.advance(count);
  };
  const changeSite = async (config?: SiteActivationRequest) => {
    const request = ++generation.current;
    defaultLoaded.current = true;
    setPlaying(false);
    setRunning(false);
    setReading(true);
    setNotice(null);
    const next = await (config ? live.activateSite(config) : live.deactivateSite());
    if (request !== generation.current) return false;
    setReading(false);
    if (!next) return false;
    liveRun.current = next.run_id;
    setRecording(null);
    setHistorical(null);
    cache.current.clear();
    return true;
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
  const activateScenario = async (previewId: string, controller: "MPC" | "H") => {
    const request = ++generation.current;
    defaultLoaded.current = true;
    setPlaying(false);
    setRunning(false);
    setReading(true);
    setNotice(null);
    const next = await live.activateScenario(previewId, controller);
    if (request !== generation.current) return false;
    setReading(false);
    if (!next) return false;
    liveRun.current = next.run_id;
    setRecording(null);
    setHistorical(null);
    cache.current.clear();
    void reloadCatalog();
    return true;
  };
  return {
    live,
    snapshot,
    siteForecast,
    time,
    frame,
    index,
    latestIndex,
    recording,
    catalog,
    catalogStatus,
    historical,
    selectedUnit,
    setSelectedUnit,
    agent,
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
    activateSite: (config: SiteActivationRequest) => changeSite(config),
    deactivateSite: () => changeSite(),
    activateScenario,
    exportRecording,
  };
}
