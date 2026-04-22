// ============================================================
// script.js
// Handles ALL browser-side behavior:
//   1. Canvas drawing (mouse + touch support)
//   2. Sending the drawn image to the Flask backend
//   3. Displaying the prediction result
// ============================================================

// ---- STEP 1: Get references to HTML elements ----
// document.getElementById() finds an element by its id attribute
// We store references so we don't have to search the DOM every time

const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");
// getContext("2d") gives us a 2D drawing interface
// ctx is like a "pen" we use to draw on the canvas

const canvasHint = document.getElementById("canvasHint");
const brushSlider = document.getElementById("brushSize");

// Result panel sections
const resultIdle = document.getElementById("resultIdle");
const resultLoading = document.getElementById("resultLoading");
const resultDisplay = document.getElementById("resultDisplay");
const resultError = document.getElementById("resultError");

// Result content elements
const predictedDigitEl = document.getElementById("predictedDigit");
const confidenceValueEl = document.getElementById("confidenceValue");
const confidenceBarEl = document.getElementById("confidenceBar");
const probabilityBarsEl = document.getElementById("probabilityBars");
const errorMessageEl = document.getElementById("errorMessage");

// ---- STEP 2: Initialize the canvas ----

// Fill canvas with black background
// (MNIST digits are white on black, so we match that)
ctx.fillStyle = "#111827";
ctx.fillRect(0, 0, canvas.width, canvas.height);

// ---- STEP 3: Drawing state variables ----
// These variables track whether the user is currently drawing

let isDrawing = false; // true while mouse button is held down
let hasDrawn = false; // true once user has drawn at least once
let lastX = 0; // X coordinate of the last mouse position
let lastY = 0; // Y coordinate of the last mouse position

// ---- STEP 4: Drawing functions ----

/**
 * startDrawing is called when the user presses the mouse button down.
 * It records the starting position and sets isDrawing = true.
 */
function startDrawing(e) {
  isDrawing = true;
  const pos = getPosition(e);
  lastX = pos.x;
  lastY = pos.y;

  // Hide the hint text once user starts drawing
  if (!hasDrawn) {
    hasDrawn = true;
    canvasHint.classList.add("hidden");
    canvas.parentElement.classList.add("active");
  }
}

/**
 * draw is called on every mouse move while isDrawing is true.
 * It draws a line from the last position to the current position.
 */
function draw(e) {
  if (!isDrawing) return; // Only draw if mouse button is held

  // Prevent page scrolling while drawing on touch devices
  e.preventDefault();

  const pos = getPosition(e);

  // ctx.beginPath() starts a new drawing path
  ctx.beginPath();

  // strokeStyle: the color of the line (white, like MNIST digits)
  ctx.strokeStyle = "#FFFFFF";

  // lineWidth: brush size from the slider
  ctx.lineWidth = parseInt(brushSlider.value);

  // lineCap: "round" makes the ends of strokes smooth and round
  ctx.lineCap = "round";

  // lineJoin: "round" makes corners smooth
  ctx.lineJoin = "round";

  // moveTo: pick up pen and move to last position (without drawing)
  ctx.moveTo(lastX, lastY);

  // lineTo: draw a line to the current position
  ctx.lineTo(pos.x, pos.y);

  // stroke(): actually render the line on the canvas
  ctx.stroke();

  // Update last position for the next movement
  lastX = pos.x;
  lastY = pos.y;
}

/**
 * stopDrawing is called when mouse button is released or mouse leaves canvas.
 */
function stopDrawing() {
  isDrawing = false;
}

/**
 * getPosition extracts the X, Y coordinates from either a mouse or touch event,
 * relative to the canvas element.
 *
 * This handles both mouse events (click/drag) and touch events (mobile).
 */
function getPosition(e) {
  // getBoundingClientRect() returns the canvas's position on the page
  const rect = canvas.getBoundingClientRect();

  // Touch events have a different structure than mouse events
  // e.touches[0] = the first finger touching the screen
  if (e.touches) {
    return {
      x: e.touches[0].clientX - rect.left,
      y: e.touches[0].clientY - rect.top,
    };
  }

  // Mouse event: clientX/clientY is the position relative to the browser window
  // Subtract rect.left/top to get position relative to the canvas
  return {
    x: e.clientX - rect.left,
    y: e.clientY - rect.top,
  };
}

// ---- STEP 5: Attach event listeners ----
// Event listeners "listen" for user actions and call our functions

// --- Mouse events ---
canvas.addEventListener("mousedown", startDrawing); // Mouse button pressed
canvas.addEventListener("mousemove", draw); // Mouse moved while pressed
canvas.addEventListener("mouseup", stopDrawing); // Mouse button released
canvas.addEventListener("mouseleave", stopDrawing); // Mouse left canvas area

// --- Touch events (for phones and tablets) ---
// { passive: false } is needed so we can call e.preventDefault() inside
canvas.addEventListener("touchstart", startDrawing, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", stopDrawing);

// ---- STEP 6: clearCanvas function ----
/**
 * Clears the canvas back to solid black.
 * Called when the user clicks the "Clear" button (onclick="clearCanvas()").
 */
function clearCanvas() {
  // Fill the entire canvas with the background color
  ctx.fillStyle = "#111827";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Reset drawing state
  hasDrawn = false;

  // Show the hint text again
  canvasHint.classList.remove("hidden");
  canvas.parentElement.classList.remove("active");

  // Reset result panel to idle state
  showSection("idle");
}

// ---- STEP 7: showSection helper ----
/**
 * Shows one of the four result panel states and hides the others.
 * @param {string} section - "idle" | "loading" | "result" | "error"
 */
function showSection(section) {
  // Hide all sections first
  resultIdle.classList.add("hidden");
  resultLoading.classList.add("hidden");
  resultDisplay.classList.add("hidden");
  resultError.classList.add("hidden");

  // Show the requested section
  if (section === "idle") resultIdle.classList.remove("hidden");
  if (section === "loading") resultLoading.classList.remove("hidden");
  if (section === "result") resultDisplay.classList.remove("hidden");
  if (section === "error") resultError.classList.remove("hidden");
}

// ---- STEP 8: predict function ----
/**
 * Called when user clicks "Predict Digit".
 * Sends the canvas image to the Flask backend and shows the result.
 */
async function predict() {
  // Check if the user has drawn anything
  if (!hasDrawn) {
    // Briefly animate the canvas to indicate "please draw first"
    canvas.parentElement.style.animation = "pulseGlow 0.4s ease 3";
    setTimeout(() => {
      canvas.parentElement.style.animation = "";
    }, 1200);
    return;
  }

  // Show loading spinner
  showSection("loading");

  try {
    // -------------------------------------------------------
    // Convert the canvas to a base64 PNG image string
    // -------------------------------------------------------
    // canvas.toDataURL() encodes the entire canvas as a PNG image
    // and returns it as a base64 string like:
    // "data:image/png;base64,iVBORw0KGgoAAAANS..."
    const imageData = canvas.toDataURL("image/png");

    // -------------------------------------------------------
    // Send the image to the Flask backend using fetch()
    // -------------------------------------------------------
    // fetch() makes an HTTP request to the given URL
    // We send a POST request with the image in JSON format
    const response = await fetch("/predict", {
      method: "POST",

      // headers tell the server we're sending JSON
      headers: {
        "Content-Type": "application/json",
      },

      // JSON.stringify converts our JavaScript object to a JSON string
      body: JSON.stringify({ image: imageData }),
    });

    // -------------------------------------------------------
    // Parse the response
    // -------------------------------------------------------
    // response.json() parses the JSON response from Flask
    // This gives us the object: { digit, confidence, all_probabilities }
    const data = await response.json();

    // Check if the backend returned an error
    if (data.error) {
      throw new Error(data.error);
    }

    // -------------------------------------------------------
    // Display the results
    // -------------------------------------------------------
    displayResult(data);
  } catch (error) {
    // If anything went wrong (network error, server error, etc.)
    console.error("Prediction failed:", error);

    // Show the error in the UI
    errorMessageEl.textContent = error.message || "Connection failed.";
    showSection("error");
  }
}

// ---- STEP 9: displayResult function ----
/**
 * Updates the result panel with the prediction data from the backend.
 * @param {Object} data - { digit, confidence, all_probabilities }
 */
function displayResult(data) {
  const { digit, confidence, all_probabilities } = data;

  // ---- Update the large digit display ----
  predictedDigitEl.textContent = digit;

  // ---- Update the confidence value text ----
  confidenceValueEl.textContent = `${confidence.toFixed(1)}%`;

  // ---- Animate the confidence bar ----
  // We set width to 0 first (in case of re-prediction), then animate
  confidenceBarEl.style.width = "0%";

  // requestAnimationFrame waits for the next screen refresh before
  // setting the new width, which triggers the CSS transition animation
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      confidenceBarEl.style.width = `${confidence}%`;
    });
  });

  // ---- Change confidence bar color based on confidence level ----
  if (confidence >= 80) {
    // High confidence: green
    confidenceBarEl.style.background =
      "linear-gradient(90deg, #059669, #10b981)";
    confidenceBarEl.style.boxShadow = "0 0 10px rgba(16, 185, 129, 0.5)";
  } else if (confidence >= 50) {
    // Medium confidence: amber/yellow
    confidenceBarEl.style.background =
      "linear-gradient(90deg, #d97706, #f59e0b)";
    confidenceBarEl.style.boxShadow = "0 0 10px rgba(245, 158, 11, 0.5)";
  } else {
    // Low confidence: red
    confidenceBarEl.style.background =
      "linear-gradient(90deg, #dc2626, #ef4444)";
    confidenceBarEl.style.boxShadow = "0 0 10px rgba(239, 68, 68, 0.5)";
  }

  // ---- Build the probability breakdown bars ----
  probabilityBarsEl.innerHTML = ""; // Clear previous bars

  // Sort digits by probability, highest first
  const sortedDigits = Object.keys(all_probabilities).sort(
    (a, b) => all_probabilities[b] - all_probabilities[a],
  );

  sortedDigits.forEach((d) => {
    const prob = all_probabilities[d];
    const isPredicted = parseInt(d) === digit;

    // Create a row for this digit
    // Template literals (backticks) let us embed variables with ${...}
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <span class="prob-digit-label">${d}</span>
      <div class="prob-bar-track">
        <div
          class="prob-bar-fill ${isPredicted ? "is-predicted" : ""}"
          style="width: 0%"
          data-target="${prob}"
        ></div>
      </div>
      <span class="prob-percent-label">${prob.toFixed(1)}%</span>
    `;

    probabilityBarsEl.appendChild(row);
  });

  // Show the result panel
  showSection("result");

  // Animate all probability bars after a brief delay
  setTimeout(() => {
    document.querySelectorAll(".prob-bar-fill").forEach((bar) => {
      bar.style.width = `${bar.dataset.target}%`;
    });
  }, 100);
}

// ---- STEP 10: Keyboard shortcut ----
// Allow pressing Enter to predict and Delete/Backspace to clear
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter") predict();
  if (e.key === "Delete" || e.key === "Backspace") clearCanvas();
});
