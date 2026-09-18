import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useWorkspace } from "./context";
import "./models.css";

export function Models() {
  const w = useWorkspace();
  const navigate = useNavigate();
  const [selection, setSelection] = useState<string | null>(null);
  const [opening, setOpening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const models = w.catalog.filter((item) =>
    item.available && item.mode === "recorded_research" &&
    item.controller === "PPO" && item.status === "complete" && item.provenance,
  );
  const selected = models.find((item) => item.id === selection) ??
    models.find((item) => item.id === w.recording?.snapshot.run_id) ??
    models.find((item) => item.featured) ?? models[0];

  const play = async () => {
    if (!selected || opening) return;
    setOpening(true);
    setError(null);
    const loaded = await w.loadRecording(selected.id);
    setOpening(false);
    if (!loaded) {
      setError("The replay couldn’t be opened. Try again.");
      return;
    }
    navigate("/plant");
    w.setPlaying(true);
  };

  return <div className="w-models">
    <div className="w-page-heading">
      <div>
        <h1>Models</h1>
        <p>Select a trained model and play its demo.</p>
      </div>
    </div>

    {models.length > 0 ? <form onSubmit={(event) => { event.preventDefault(); void play(); }}>
      <fieldset className="w-model-options" aria-label="Model" disabled={opening || w.busy}>
        {models.map((model) => <label className="w-model-option" key={model.id}>
          <input type="radio" name="model" value={model.id} checked={selected?.id === model.id}
            onChange={() => { setSelection(model.id); setError(null); }} />
          <span>
            <strong>PPO {model.provenance!.optimizer_seed}</strong>
            <small>Trained plant controller · {model.provenance!.mission.horizon_h / 24}-day replay</small>
          </span>
        </label>)}
      </fieldset>
      <button className="w-button w-button--primary w-model-play" type="submit" disabled={opening || w.busy}>
        {opening ? "Opening replay…" : "Play replay"}
      </button>
      {error && <p className="w-model-message" role="alert">{error}</p>}
    </form> : w.catalogStatus === "loading" ? <p className="w-note" role="status">Loading models…</p> : <div>
      <p className="w-note" role={w.catalogStatus === "error" ? "alert" : "status"}>
        {w.catalogStatus === "error" ? "Models couldn’t be loaded. Try again." : "No model replays are available."}
      </p>
      <button className="w-button" onClick={() => void w.reloadCatalog()}>Try again</button>
    </div>}
  </div>;
}
