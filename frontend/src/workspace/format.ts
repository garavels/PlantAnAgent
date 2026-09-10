export const n = (value: number, digits = 1) =>
  (Math.abs(value) < 0.5 * 10 ** -digits ? 0 : value).toLocaleString("en-GB", {
    maximumFractionDigits: digits,
  });
export const eur = (value: number) =>
  `${Math.round(value * 100) < 0 ? "−" : ""}€${n(Math.abs(value), 2)}`;
export const duration = (hours: number) => {
  const m = Math.max(0, Math.round(hours * 60));
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
};
export const time = (hours: number) => {
  const m = Math.round((hours + 6) * 60);
  return `D${Math.floor(m / 1440) + 1} ${String(Math.floor(m / 60) % 24).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
};
export const words = (value: string) => value.replaceAll("_", " ");
