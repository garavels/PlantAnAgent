import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { WorkspaceProvider } from "./workspace/WorkspaceProvider";
import { WorkspaceShell } from "./workspace/WorkspaceShell";
import { Models } from "./workspace/Models";
import { Site } from "./workspace/Site";
import { Scenarios } from "./workspace/Scenarios";
import {
  Overview,
  Plant,
  Weather,
  Planning,
  Deliveries,
  Economics,
  Events,
} from "./workspace/Pages";
import "./workspace/workspace.css";
export default function App() {
  return (
    <BrowserRouter useTransitions={false}>
      <WorkspaceProvider>
        <WorkspaceShell>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/plant" element={<Plant />} />
            <Route path="/weather" element={<Weather />} />
            <Route path="/site" element={<Site />} />
            <Route path="/planning" element={<Planning />} />
            <Route path="/deliveries" element={<Deliveries />} />
            <Route path="/economics" element={<Economics />} />
            <Route path="/events" element={<Events />} />
            <Route path="/models" element={<Models />} />
            <Route path="/learning" element={<Navigate to="/models" replace />} />
            <Route path="/evidence" element={<Navigate to="/models" replace />} />
            <Route path="/v2" element={<Navigate to="/" replace />} />
            <Route
              path="/plant-3d"
              element={<Navigate to="/plant" replace />}
            />
            <Route
              path="/agent"
              element={<Navigate to="/models" replace />}
            />
            <Route
              path="/metrics"
              element={<Navigate to="/economics" replace />}
            />
            <Route path="/scenarios" element={<Scenarios />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </WorkspaceShell>
      </WorkspaceProvider>
    </BrowserRouter>
  );
}
