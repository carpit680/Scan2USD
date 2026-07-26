import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import { ProjectProvider } from "./state/ProjectContext";
import { HelpProvider } from "./state/HelpContext";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <ProjectProvider>
        <HelpProvider>
          <App />
        </HelpProvider>
      </ProjectProvider>
    </BrowserRouter>
  </React.StrictMode>,
);
