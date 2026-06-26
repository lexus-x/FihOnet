// index.js — FishOnet Interactive Logic

document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupSimulator();
});

// Tab Switching
function setupTabs() {
  const buttons = document.querySelectorAll('.nav-btn');
  const contents = document.querySelectorAll('.tab-content');

  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');
      
      // Update buttons
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      // Update content
      contents.forEach(c => c.classList.remove('active'));
      document.getElementById(tabId).classList.add('active');
    });
  });
}

// Simulator Logic
const mockSamples = [
  {
    id: 1,
    common: "Ocellaris Clownfish",
    scientific: "Amphiprion ocellaris",
    type: "seen", // Seen in training
    description: "Bright orange fish with white bands outlined in black. Found in warm water reefs.",
    maxLogit: 7.25, // Score from seen classifier
    correctClass: "Ocellaris Clownfish",
    closedSetPred: "Ocellaris Clownfish",
    zeroShotPred: "Ocellaris Clownfish"
  },
  {
    id: 2,
    common: "Blue Tang",
    scientific: "Paracanthurus hepatus",
    type: "seen",
    description: "Vibrant royal blue body, yellow tail, and black 'palette' markings on the side.",
    maxLogit: 5.10,
    correctClass: "Blue Tang",
    closedSetPred: "Blue Tang",
    zeroShotPred: "Pacific Surgeonfish"
  },
  {
    id: 3,
    common: "Great White Shark",
    scientific: "Carcharodon carcharias",
    type: "seen",
    description: "Large predatory shark with a torpedo-shaped body, dark grey upper body, and white underbelly.",
    maxLogit: 8.40,
    correctClass: "Great White Shark",
    closedSetPred: "Great White Shark",
    zeroShotPred: "Basking Shark"
  },
  {
    id: 4,
    common: "Leafy Seadragon",
    scientific: "Phycodurus eques",
    type: "unseen", // Unseen/Novel in training
    description: "Ornate, leaf-like appendages over its body that act as camouflage mimicking seaweed.",
    maxLogit: 4.85, // Low similarity to any seen fish
    correctClass: "Leafy Seadragon",
    closedSetPred: "Weedy Seadragon", // Wrong prediction if closed-set
    zeroShotPred: "Leafy Seadragon" // Correct if routed to zero-shot
  },
  {
    id: 5,
    common: "Psychedelic Mandarinfish",
    scientific: "Synchiropus picturatus",
    type: "unseen",
    description: "Small, brightly colored dragonet with distinct target-shaped green, orange, and blue patterns.",
    maxLogit: 3.20,
    correctClass: "Psychedelic Mandarinfish",
    closedSetPred: "Splendid Mandarinfish",
    zeroShotPred: "Psychedelic Mandarinfish"
  },
  {
    id: 6,
    common: "Whale Shark",
    scientific: "Rhincodon typus",
    type: "unseen",
    description: "Slow-moving, filter-feeding carpet shark and the largest known extant fish species, dotted with white spots.",
    maxLogit: 5.80, // Medium-high score (shares some features with basking shark)
    correctClass: "Whale Shark",
    closedSetPred: "Basking Shark",
    zeroShotPred: "Whale Shark"
  }
];

function setupSimulator() {
  const slider = document.getElementById('threshold-slider');
  const thresholdVal = document.getElementById('threshold-val');
  
  if (!slider) return;

  const updateSimulation = () => {
    const theta = parseFloat(slider.value);
    thresholdVal.innerText = theta.toFixed(2);
    
    let correctCount = 0;
    let knownCorrect = 0;
    let knownTotal = 0;
    let unknownCorrect = 0;
    let unknownTotal = 0;
    let falsePositives = 0; // Unseen classified as Seen

    const container = document.getElementById('samples-container');
    container.innerHTML = '';

    mockSamples.forEach(sample => {
      const isKnownByGate = sample.maxLogit >= theta;
      let classificationResult = "";
      let cssClass = "";
      let predictedClass = "";

      if (sample.type === 'seen') {
        knownTotal++;
        if (isKnownByGate) {
          // Correctly routed to closed set
          predictedClass = sample.closedSetPred;
          const isAccurate = (predictedClass === sample.correctClass);
          if (isAccurate) {
            classificationResult = "Correct Known (Seen)";
            cssClass = "correct-known";
            correctCount++;
            knownCorrect++;
          } else {
            classificationResult = "Incorrect Known (Seen)";
            cssClass = "false-positive"; // effectively a classification error
          }
        } else {
          // Falsely routed to open set (False Negative for gating)
          predictedClass = sample.zeroShotPred + " (Zero-shot)";
          classificationResult = "False Negative (Seen as Unknown)";
          cssClass = "false-negative";
        }
      } else {
        // Unseen/Novel
        unknownTotal++;
        if (isKnownByGate) {
          // Falsely routed to closed-set (False Positive)
          predictedClass = sample.closedSetPred;
          classificationResult = "False Positive (Unknown as Known)";
          cssClass = "false-positive";
          falsePositives++;
        } else {
          // Correctly routed to zero-shot text classifier
          predictedClass = sample.zeroShotPred;
          const isAccurate = (predictedClass === sample.correctClass);
          if (isAccurate) {
            classificationResult = "Correct Unknown (Novel)";
            cssClass = "correct-unknown";
            correctCount++;
            unknownCorrect++;
          } else {
            classificationResult = "Incorrect Unknown (Novel)";
            cssClass = "false-negative";
          }
        }
      }

      // Create card HTML
      const item = document.createElement('div');
      item.className = `fish-item ${cssClass}`;
      item.innerHTML = `
        <div class="fish-info">
          <h4>${sample.common} <span class="badge ${sample.type === 'seen' ? 'badge-seen' : 'badge-unseen'}">${sample.type}</span></h4>
          <div class="fish-sci">${sample.scientific}</div>
          <div class="fish-desc">${sample.description}</div>
        </div>
        <div class="fish-status">
          <span class="badge-result bg-${cssClass}">${classificationResult}</span>
          <div class="score-lbl">Max Logit: <span>${sample.maxLogit.toFixed(2)}</span></div>
          <div class="score-lbl">Decision: <span>${isKnownByGate ? 'KNOWN' : 'UNKNOWN'}</span></div>
          <div class="score-lbl">Prediction: <span>${predictedClass}</span></div>
        </div>
      `;
      container.appendChild(item);
    });

    // Update Stats
    const totalAccuracy = (correctCount / mockSamples.length) * 100;
    const knownAccuracy = knownTotal > 0 ? (knownCorrect / knownTotal) * 100 : 0;
    const unknownAccuracy = unknownTotal > 0 ? (unknownCorrect / unknownTotal) * 100 : 0;
    const fpr = unknownTotal > 0 ? (falsePositives / unknownTotal) * 100 : 0;

    document.getElementById('metric-acc').innerText = `${totalAccuracy.toFixed(1)}%`;
    document.getElementById('metric-known').innerText = `${knownAccuracy.toFixed(1)}%`;
    document.getElementById('metric-unknown').innerText = `${unknownAccuracy.toFixed(1)}%`;
    document.getElementById('metric-fpr').innerText = `${fpr.toFixed(1)}%`;
  };

  slider.addEventListener('input', updateSimulation);
  updateSimulation(); // Initial call
}
