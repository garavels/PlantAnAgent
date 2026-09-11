import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { WorkspaceProvider } from "./workspace/WorkspaceProvider";
import { WorkspaceShell } from "./workspace/WorkspaceShell";
import { Learning } from "./workspace/Learning";
import {
  Overview,
  Plant,
  Weather,
  Planning,
  Deliveries,
  Economics,
  Events,
  Evidence,
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
            <Route path="/planning" element={<Planning />} />
            <Route path="/deliveries" element={<Deliveries />} />
            <Route path="/economics" element={<Economics />} />
            <Route path="/events" element={<Events />} />
            <Route path="/learning" element={<Learning />} />
            <Route path="/evidence" element={<Evidence />} />
            <Route path="/v2" element={<Navigate to="/" replace />} />
            <Route
              path="/plant-3d"
              element={<Navigate to="/plant" replace />}
            />
            <Route
              path="/agent"
              element={<Navigate to="/evidence" replace />}
            />
            <Route
              path="/metrics"
              element={<Navigate to="/economics" replace />}
            />
            <Route path="/scenarios" element={<Navigate to="/" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </WorkspaceShell>
      </WorkspaceProvider>
    </BrowserRouter>
  );
}
