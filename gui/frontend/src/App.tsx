import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { ProjectPage } from "./pages/ProjectPage";
import { PipelinePage } from "./pages/PipelinePage";
import { ConfigPage } from "./pages/ConfigPage";
import { CommandsPage } from "./pages/CommandsPage";
import { ReviewPage } from "./pages/ReviewPage";
import { MetricScalePage } from "./pages/MetricScalePage";
import { PreviewPage } from "./pages/PreviewPage";
import { QualityPage } from "./pages/QualityPage";
import { TuningPage } from "./pages/TuningPage";
import { ArtifactsPage } from "./pages/ArtifactsPage";
import { DoctorPage } from "./pages/DoctorPage";
import { GuidePage } from "./pages/GuidePage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<ProjectPage />} />
        <Route path="pipeline" element={<PipelinePage />} />
        <Route path="config" element={<ConfigPage />} />
        <Route path="commands" element={<CommandsPage />} />
        <Route path="review" element={<ReviewPage />} />
        <Route path="metric" element={<MetricScalePage />} />
        <Route path="preview" element={<PreviewPage />} />
        <Route path="quality" element={<QualityPage />} />
        <Route path="tuning" element={<TuningPage />} />
        <Route path="artifacts" element={<ArtifactsPage />} />
        <Route path="doctor" element={<DoctorPage />} />
        <Route path="guide" element={<GuidePage />} />
      </Route>
    </Routes>
  );
}
