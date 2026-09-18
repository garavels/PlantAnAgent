import type { V2DemoFrame } from "../types/v2";
import { siteTime, time } from "./format";

/** Inspector components also render without workspace context in saved frame views. */
export function equipmentTime(frame: V2DemoFrame, hours: number) {
  const utc = frame.clock.time_utc ?? frame.weather.site_sample?.time;
  return frame.weather.reference_kind === "forecast_pv" && utc
    ? siteTime(utc, hours - frame.clock.elapsed_hours)
    : time(hours);
}
