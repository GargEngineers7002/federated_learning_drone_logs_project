// document.addEventListener("DOMContentLoaded", () => {
//   const form = document.getElementById("upload-form");
//   const submitBtn = document.getElementById("submit-btn");
//   const resultsContainer = document.getElementById("results-container");
//   const plotContainer = document.getElementById("plot-container");
//   const metricsTableBody = document.getElementById("metrics-table-body");
//   const plotLoader = document.getElementById("plot-loader");
//   const metricsLoader = document.getElementById("metrics-loader");

//   form.addEventListener("submit", async (event) => {
//     event.preventDefault(); // Prevent default form submission

//     // Show loaders and hide previous results
//     resultsContainer.classList.remove("hidden");
//     plotLoader.classList.remove("hidden");
//     metricsLoader.classList.remove("hidden");
//     Plotly.purge(plotContainer); // Clear previous plot
//     metricsTableBody.innerHTML = ""; // Clear previous metrics
//     submitBtn.disabled = true;
//     submitBtn.textContent = "Processing...";

//     // Use FormData to send both the file and the select value
//     const formData = new FormData(form);

//     try {
//       const response = await fetch("/api/predict_trajectory", {
//         method: "POST",
//         body: formData,
//       });

//       if (!response.ok) {
//         const errorData = await response.json();
//         throw new Error(
//           errorData.detail || `HTTP error! Status: ${response.status}`
//         );
//       }

//       const data = await response.json();
//       displayResults(data.results);
//     } catch (error) {
//       console.error("Error:", error);
//       alert(`An error occurred: ${error.message}`);
//     } finally {
//       // Hide loaders and re-enable button
//       plotLoader.classList.add("hidden");
//       metricsLoader.classList.add("hidden");
//       submitBtn.disabled = false;
//       submitBtn.textContent = "Predict Trajectory";
//     }
//   });

//   function displayResults(results) {
//     const plotData = [];
//     const modelNames = Object.keys(results);

//     // --- 1. Populate Plot Data ---
//     modelNames.forEach((modelName) => {
//       const modelResult = results[modelName];
//       const trace = {
//         x: modelResult.trajectory.x,
//         y: modelResult.trajectory.y,
//         z: modelResult.trajectory.z,
//         mode: "lines",
//         type: "scatter3d",
//         name: modelName,
//         line: { width: 4 },
//       };
//       plotData.push(trace);
//     });

//     const layout = {
//       title: "3D Trajectory Comparison",
//       scene: {
//         xaxis: { title: "X Coordinate" },
//         yaxis: { title: "Y Coordinate" },
//         zaxis: { title: "Z Coordinate" },
//       },
//       margin: { l: 0, r: 0, b: 0, t: 40 },
//     };

//     Plotly.newPlot(plotContainer, plotData, layout);

//     // --- 2. Populate Metrics Table ---
//     metricsTableBody.innerHTML = ""; // Clear old data
//     modelNames.forEach((modelName) => {
//       const metrics = results[modelName].metrics;
//       const row = document.createElement("tr");
//       row.innerHTML = `
//                 <td>${modelName}</td>
//                 <td>${metrics.RMSE}</td>
//                 <td>${metrics.MAE}</td>
//             `;
//       metricsTableBody.appendChild(row);
//     });
//   }
// });



document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predictForm"); 
  const submitBtn = document.getElementById("submit-btn");
  const resultsContainer = document.getElementById("results-container");

  const plot2D = document.getElementById("plot-2d");
  const plot3D = document.getElementById("plot-container");

  const metricsTableBody = document.getElementById("metrics-table-body");
  const plotLoader = document.getElementById("plot-loader");
  const metricsLoader = document.getElementById("metrics-loader");

  if (!form) {
    console.error("Form not found");
    return;
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    resultsContainer.classList.remove("hidden");
    plotLoader.classList.remove("hidden");
    metricsLoader.classList.remove("hidden");

    Plotly.purge(plot2D);
    Plotly.purge(plot3D);
    metricsTableBody.innerHTML = "";

    submitBtn.disabled = true;
    submitBtn.textContent = "Processing...";

    const formData = new FormData(form);

    try {
      const response = await fetch("/api/predict_trajectory", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();
      displayResults(data.results);
    } catch (error) {
      console.error(error);
      alert(error.message);
    } finally {
      plotLoader.classList.add("hidden");
      metricsLoader.classList.add("hidden");
      submitBtn.disabled = false;
      submitBtn.textContent = "Predict Trajectory";
    }
  });

function displayResults(results) {
    if (!results) return;

    const metricsTableBody = document.getElementById("metrics-table-body");

    // ---- Extract actual trajectory (ground truth) ----
    const actual = results.actual_trajectory || null;

    // ---- Find the Best Model (lowest RMSE) ----
    // Filter out non-model keys like "actual_trajectory"
    const modelNames = Object.keys(results).filter(k => k !== "actual_trajectory" && results[k].trajectory);
    const sortedModels = modelNames.sort((a, b) => {
      const ra = parseFloat(results[a].metrics.RMSE) || Infinity;
      const rb = parseFloat(results[b].metrics.RMSE) || Infinity;
      return ra - rb;
    });

    const bestModelName = sortedModels[0];
    if (!bestModelName) return;

    const bestTraj = results[bestModelName].trajectory;

    // =============================================
    //  2D PLOT — Longitude vs Latitude
    // =============================================
    const plot2DData = [];

    // Actual trajectory (blue, dashed line + circles)
    if (actual) {
      plot2DData.push({
        x: actual.x,
        y: actual.y,
        mode: "lines+markers",
        type: "scatter",
        name: "Actual Trajectory",
        marker: { size: 6, color: "#1565C0", symbol: "circle", opacity: 0.8 },
        line:   { color: "#1565C0", width: 3, dash: "dot" }
      });
    }

    // Predicted trajectory (red, solid line + diamonds)
    plot2DData.push({
      x: bestTraj.x,
      y: bestTraj.y,
      mode: "lines+markers",
      type: "scatter",
      name: `Predicted — ${bestModelName}`,
      marker: { size: 3, color: "#E53935", symbol: "diamond", opacity: 1 },
      line:   { color: "#E53935", width: 1.5 }
    });

    const layout2D = {
      title: {
        text: `<b>2D Trajectory — ${bestModelName}</b>`,
        font: { family: "Inter, sans-serif", size: 16 }
      },
      font: { family: "Inter, sans-serif", color: "#333", size: 12 },
      autosize: true,
      xaxis: {
        title: { text: "<b>Longitude</b>", font: { size: 14 }, standoff: 15 },
        tickformat: ".6f",
        tickangle: -45,
        mirror: true,
        linecolor: "#999",
        linewidth: 1,
        showgrid: true,
        gridcolor: "#eee",
        zeroline: false
      },
      yaxis: {
        title: { text: "<b>Latitude</b>", font: { size: 14 }, standoff: 15 },
        tickformat: ".6f",
        mirror: true,
        linecolor: "#999",
        linewidth: 1,
        showgrid: true,
        gridcolor: "#eee",
        zeroline: false
      },
      margin: { l: 90, r: 40, b: 110, t: 60 },
      showlegend: true,
      legend: {
        x: 0.01, y: 0.99,
        bgcolor: "rgba(255,255,255,0.85)",
        bordercolor: "#ccc",
        borderwidth: 1,
        font: { size: 11 }
      },
      plot_bgcolor: "#fafafa"
    };

    Plotly.newPlot("plot-2d", plot2DData, layout2D);

    // =============================================
    //  3D PLOT — Longitude, Latitude, Altitude
    // =============================================
    const plot3DData = [];

    // Actual trajectory (blue)
    if (actual) {
      plot3DData.push({
        x: actual.x,
        y: actual.y,
        z: actual.z,
        mode: "lines",
        type: "scatter3d",
        name: "Actual Trajectory",
        line: { color: "#1565C0", width: 4, dash: "dot" }
      });
    }

    // Predicted trajectory (red/green)
    plot3DData.push({
      x: bestTraj.x,
      y: bestTraj.y,
      z: bestTraj.z,
      mode: "lines",
      type: "scatter3d",
      name: `Predicted — ${bestModelName}`,
      line: { color: "#E53935", width: 4 }
    });

    Plotly.newPlot("plot-container", plot3DData, {
      title: {
        text: `<b>3D Trajectory — ${bestModelName}</b>`,
        font: { family: "Inter, sans-serif", size: 16 }
      },
      scene: {
        xaxis: { title: "Longitude" },
        yaxis: { title: "Latitude" },
        zaxis: { title: "Altitude" },
      },
      showlegend: true,
      legend: {
        x: 0.01, y: 0.99,
        bgcolor: "rgba(255,255,255,0.85)",
        bordercolor: "#ccc",
        borderwidth: 1,
        font: { size: 11 }
      },
      margin: { l: 0, r: 0, b: 0, t: 40 }
    });

    // =============================================
    //  METRICS TABLE — All models, best highlighted
    // =============================================
    if (metricsTableBody) {
      metricsTableBody.innerHTML = "";

      sortedModels.forEach((modelName) => {
        const m = results[modelName].metrics;
        const isBest = modelName === bestModelName;

        const rowStyle = isBest ? 'style="background-color: #e8f5e9; font-weight: bold;"' : '';
        const icon = isBest ? '🏆 ' : '';

        metricsTableBody.innerHTML += `
          <tr ${rowStyle}>
            <td>${icon}${modelName}</td>
            <td>${m.RMSE}</td>
            <td>${m.MAE}</td>
          </tr>`;
      });
    }
  }
});
