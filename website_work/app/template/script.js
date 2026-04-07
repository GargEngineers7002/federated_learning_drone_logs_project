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

    try {
      console.log("Starting FL process...");
      // Use CPU backend for training to avoid WebGL/GPU initialization slowness with large LSTMs
      await tf.setBackend('cpu');
      console.log("Using backend:", tf.getBackend());

      const gFormData = new FormData();
      gFormData.append("uav_model", formData.get("uav_model"));
      const response_global_model = await fetch("/api/get_global", {
        method: "POST",
        body: gFormData,
      });

      if (!response_global_model.ok) {
        const errorData = await response_global_model.json();
        throw new Error(
          errorData.detail || `HTTP ${response_global_model.status}`,
        );
      }

      const global_data = await response_global_model.json();
      const weights = global_data.weights;
      const config = global_data.config;

      if (!weights || weights.length === 0) {
        throw new Error("Invalid weights received");
      }

      console.log("Reconstructing model from config...");

      const response_process_data = await fetch("/api/get_processed", {
        method: "POST",
        body: formData,
      });

      if (!response_process_data.ok) {
        const errorData = await response_process_data.json();
        throw new Error(
          errorData.detail || `HTTP ${response_process_data.status}`,
        );
      }

      const processed_data_raw = await response_process_data.json();

      // Reconstruct model from Keras config
      const model = tf.sequential();
      
      // Determine expected input features from weights[0]
      const input_features = weights[0].shape ? weights[0].shape[0] : weights[0].length;
      
      const actual_features = processed_data_raw.x[0][0].length;
      
      let x_train_data = processed_data_raw.x;
      let y_train_data = processed_data_raw.y;

      // Limit training data to last 500 samples for responsiveness
      const MAX_SAMPLES = 500;
      if (x_train_data.length > MAX_SAMPLES) {
          console.log(`Limiting training data from ${x_train_data.length} to ${MAX_SAMPLES} samples.`);
          x_train_data = x_train_data.slice(-MAX_SAMPLES);
          y_train_data = y_train_data.slice(-MAX_SAMPLES);
      }

      if (actual_features > input_features) {
          console.warn(`Feature mismatch: expected ${input_features}, got ${actual_features}. Trimming first column...`);
          x_train_data = x_train_data.map(seq => seq.map(row => row.slice(-input_features)));
      }

      const x_train = tf.tensor3d(x_train_data);
      const y_train = tf.tensor2d(y_train_data);

      // Rebuild layers based on config
      config.layers.forEach((layerConfig, index) => {
        if (layerConfig.class_name === "InputLayer") return;
        
        let layer;
        const props = layerConfig.config;
        
        // Use 'zeros' initializers because we will overwrite them immediately with model.setWeights
        // This avoids the slow 'Orthogonal' initialization warning/delay.
        switch (layerConfig.class_name) {
          case "LSTM":
            layer = tf.layers.lstm({
              units: props.units,
              returnSequences: props.return_sequences,
              useBias: props.use_bias !== undefined ? props.use_bias : true,
              kernelInitializer: 'zeros',
              recurrentInitializer: 'zeros',
              biasInitializer: 'zeros',
              inputShape: index === 1 ? [x_train_data[0].length, input_features] : undefined
            });
            break;
          case "BatchNormalization":
            layer = tf.layers.batchNormalization({ 
                axis: props.axis,
                center: props.center !== undefined ? props.center : true,
                scale: props.scale !== undefined ? props.scale : true
            });
            break;
          case "Dropout":
            layer = tf.layers.dropout({ rate: props.rate });
            break;
          case "Dense":
            layer = tf.layers.dense({ 
                units: props.units, 
                activation: props.activation,
                kernelInitializer: 'zeros',
                biasInitializer: 'zeros',
                useBias: props.use_bias !== undefined ? props.use_bias : true
            });
            break;
        }
        if (layer) model.add(layer);
      });

      // Reorder weights: Trainable first, then Non-Trainable (to match TF.js Sequential)
      const trainableWeights = [];
      const nonTrainableWeights = [];
      let weightIdx = 0;

      config.layers.forEach((layerConfig) => {
        if (layerConfig.class_name === "InputLayer") return;
        const props = layerConfig.config;
        
        if (layerConfig.class_name === "LSTM") {
          const numWeights = (props.use_bias !== false) ? 3 : 2;
          for (let i = 0; i < numWeights; i++) trainableWeights.push(weights[weightIdx++]);
        } else if (layerConfig.class_name === "BatchNormalization") {
          // Keras BN order: [gamma, beta, moving_mean, moving_variance]
          // gamma/beta are trainable, mean/variance are not.
          if (props.scale !== false) trainableWeights.push(weights[weightIdx++]);
          if (props.center !== false) trainableWeights.push(weights[weightIdx++]);
          nonTrainableWeights.push(weights[weightIdx++]); // moving_mean
          nonTrainableWeights.push(weights[weightIdx++]); // moving_variance
        } else if (layerConfig.class_name === "Dense") {
          const numWeights = (props.use_bias !== false) ? 2 : 1;
          for (let i = 0; i < numWeights; i++) trainableWeights.push(weights[weightIdx++]);
        } else if (layerConfig.class_name === "Dropout") {
          // No weights
        }
      });
      
      const reorderedWeights = trainableWeights.concat(nonTrainableWeights);

      // Set weights
      console.log("Setting model weights...");
      const tensors = reorderedWeights.map((w, i) => {
          const tensor = tf.tensor(w);
          return tensor;
      });
      
      try {
          model.setWeights(tensors);
      } catch (e) {
          console.error("Weight Loading Error details:");
          model.weights.forEach((w, i) => {
              console.log(`Layer weight ${i} (${w.name}): expected shape ${w.shape}, provided shape ${tensors[i].shape}`);
          });
          throw e;
      }
      tensors.forEach((t) => t.dispose());

      model.compile({
        optimizer: tf.train.adam(),
        loss: "meanSquaredError",
      });

      console.log("training model...");
      await model.fit(x_train, y_train, {
        epochs: 1,
        batchSize: 32,
        verbose: 0,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch + 1} done. Loss: ${logs.loss.toFixed(6)}`);
            }
        }
      });

      console.log("training done");

      // Extract weights and re-align them to original Keras order
      const trainedWeights = model.getWeights();
      const numTrainable = trainableWeights.length;
      
      const newTrainable = trainedWeights.slice(0, numTrainable);
      const newNonTrainable = trainedWeights.slice(numTrainable);
      
      const kerasWeights = new Array(weights.length);
      let tIdx = 0;
      let ntIdx = 0;
      let wIdx = 0;

      // Use the same config-based logic to put weights back in Keras order
      config.layers.forEach((layerConfig) => {
          if (layerConfig.class_name === "InputLayer") return;
          const props = layerConfig.config;
          
          if (layerConfig.class_name === "LSTM") {
            const count = (props.use_bias !== false) ? 3 : 2;
            for (let i = 0; i < count; i++) kerasWeights[wIdx++] = newTrainable[tIdx++].arraySync();
          } else if (layerConfig.class_name === "BatchNormalization") {
            if (props.scale !== false) kerasWeights[wIdx++] = newTrainable[tIdx++].arraySync();
            if (props.center !== false) kerasWeights[wIdx++] = newTrainable[tIdx++].arraySync();
            kerasWeights[wIdx++] = newNonTrainable[ntIdx++].arraySync(); // moving_mean
            kerasWeights[wIdx++] = newNonTrainable[ntIdx++].arraySync(); // moving_variance
          } else if (layerConfig.class_name === "Dense") {
            const count = (props.use_bias !== false) ? 2 : 1;
            for (let i = 0; i < count; i++) kerasWeights[wIdx++] = newTrainable[tIdx++].arraySync();
          }
      });

      trainedWeights.forEach(t => t.dispose());

      // Dispose training data and model
      x_train.dispose();
      y_train.dispose();
      model.dispose();

      // Return for federated averaging
      const flFormData = new FormData();
      flFormData.append("uav_model", formData.get("uav_model"));
      flFormData.append("weights", JSON.stringify(kerasWeights));

      const flResponse = await fetch("/api/federated_averaging", {
        method: "POST",
        body: flFormData,
      });

      if (!flResponse.ok) {
        const errorData = await flResponse.json();
        throw new Error(errorData.detail || `HTTP ${flResponse.status}`);
      }

      console.log("federated averaging data sent");
    } catch (error) {
      console.error("FL Error:", error);
    }
  });

  function displayResults(results) {
    if (!results) return;

    const metricsTableBody = document.getElementById("metrics-table-body");

    // ---- Extract actual trajectory (ground truth) ----
    const actual = results.actual_trajectory || null;

    // ---- Find the Best Model (lowest RMSE) ----
    // Filter out non-model keys like "actual_trajectory"
    const modelNames = Object.keys(results).filter(
      (k) => k !== "actual_trajectory" && results[k].trajectory,
    );
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
        line: { color: "#1565C0", width: 3, dash: "dot" },
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
      line: { color: "#E53935", width: 1.5 },
    });

    const layout2D = {
      title: {
        text: `<b>2D Trajectory — ${bestModelName}</b>`,
        font: { family: "Inter, sans-serif", size: 16 },
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
        zeroline: false,
      },
      yaxis: {
        title: { text: "<b>Latitude</b>", font: { size: 14 }, standoff: 15 },
        tickformat: ".6f",
        mirror: true,
        linecolor: "#999",
        linewidth: 1,
        showgrid: true,
        gridcolor: "#eee",
        zeroline: false,
      },
      margin: { l: 90, r: 40, b: 110, t: 60 },
      showlegend: true,
      legend: {
        x: 0.01,
        y: 0.99,
        bgcolor: "rgba(255,255,255,0.85)",
        bordercolor: "#ccc",
        borderwidth: 1,
        font: { size: 11 },
      },
      plot_bgcolor: "#fafafa",
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
        line: { color: "#1565C0", width: 4, dash: "dot" },
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
      line: { color: "#E53935", width: 4 },
    });

    Plotly.newPlot("plot-container", plot3DData, {
      title: {
        text: `<b>3D Trajectory — ${bestModelName}</b>`,
        font: { family: "Inter, sans-serif", size: 16 },
      },
      scene: {
        xaxis: { title: "Longitude" },
        yaxis: { title: "Latitude" },
        zaxis: { title: "Altitude" },
      },
      showlegend: true,
      legend: {
        x: 0.01,
        y: 0.99,
        bgcolor: "rgba(255,255,255,0.85)",
        bordercolor: "#ccc",
        borderwidth: 1,
        font: { size: 11 },
      },
      margin: { l: 0, r: 0, b: 0, t: 40 },
    });

    // =============================================
    //  METRICS TABLE — All models, best highlighted
    // =============================================
    if (metricsTableBody) {
      metricsTableBody.innerHTML = "";

      sortedModels.forEach((modelName) => {
        const m = results[modelName].metrics;
        const isBest = modelName === bestModelName;

        const rowStyle = isBest
          ? 'style="background-color: #e8f5e9; font-weight: bold;"'
          : "";
        const icon = isBest ? "🏆 " : "";

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
