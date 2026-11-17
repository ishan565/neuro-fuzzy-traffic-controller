/**
 * Smart Traffic Signal Control — Dashboard Logic
 * ================================================
 * Implements:
 *   - JavaScript-based Fuzzy Inference (simplified for browser)
 *   - Traffic intersection simulation with animated vehicles
 *   - Real-time neuro-fuzzy output computation
 *   - Interactive control panel
 *   - Active rule highlighting
 */

// ============================================================
// SIMPLIFIED FUZZY INFERENCE SYSTEM (Browser-side)
// ============================================================

/**
 * Triangular membership function.
 * @param {number} x - Input value
 * @param {number} a - Left foot
 * @param {number} b - Peak
 * @param {number} c - Right foot
 * @returns {number} Membership degree [0, 1]
 */
function trimf(x, a, b, c) {
    if (x <= a || x >= c) return 0;
    if (x <= b) return (x - a) / (b - a);
    return (c - x) / (c - b);
}

/**
 * Trapezoidal membership function.
 */
function trapmf(x, a, b, c, d) {
    if (x <= a || x >= d) return 0;
    if (x >= b && x <= c) return 1;
    if (x < b) return (x - a) / (b - a);
    return (d - x) / (d - c);
}

// Membership functions for inputs
const MF = {
    vehicles: {
        low: (x) => trimf(x, 0, 0, 35),
        medium: (x) => trimf(x, 20, 50, 80),
        high: (x) => trimf(x, 65, 100, 100)
    },
    density: {
        low: (x) => trimf(x, 0, 0, 35),
        medium: (x) => trimf(x, 20, 50, 80),
        high: (x) => trimf(x, 65, 100, 100)
    },
    waiting: {
        short: (x) => trimf(x, 0, 0, 60),
        medium: (x) => trimf(x, 30, 90, 150),
        long: (x) => trimf(x, 120, 180, 180)
    },
    emergency: {
        no: (x) => trapmf(x, 0, 0, 0.3, 0.5),
        yes: (x) => trapmf(x, 0.5, 0.7, 1, 1)
    }
};

// Output centroids (simplified defuzzification)
const OUTPUT = {
    green: { very_short: 15, short: 35, medium: 60, long: 85, very_long: 110 },
    priority: { low: 1, medium: 2.5, high: 3.5, very_high: 4.5, emergency: 5 },
    congestion: { low: 10, moderate: 40, high: 70, severe: 90 }
};

// Fuzzy rules
const FUZZY_RULES = [
    // Emergency
    { id: 1, cond: (v, d, w, e) => MF.emergency.yes(e),
      out: { green: 'very_long', priority: 'emergency', congestion: 'severe' },
      text: 'IF emergency = YES → green = Very Long, priority = Emergency' },
    
    // High density
    { id: 2, cond: (v, d, w, e) => Math.min(MF.density.high(d), MF.waiting.long(w), MF.vehicles.high(v)),
      out: { green: 'very_long', priority: 'very_high', congestion: 'severe' },
      text: 'IF density = High AND wait = Long AND vehicles = High → green = Very Long' },
    { id: 3, cond: (v, d, w, e) => Math.min(MF.density.high(d), MF.waiting.long(w), MF.vehicles.medium(v)),
      out: { green: 'long', priority: 'very_high', congestion: 'severe' },
      text: 'IF density = High AND wait = Long AND vehicles = Medium → green = Long' },
    { id: 4, cond: (v, d, w, e) => Math.min(MF.density.high(d), MF.waiting.medium(w), MF.vehicles.high(v)),
      out: { green: 'long', priority: 'very_high', congestion: 'severe' },
      text: 'IF density = High AND wait = Medium AND vehicles = High → green = Long' },
    { id: 5, cond: (v, d, w, e) => Math.min(MF.density.high(d), MF.waiting.medium(w), MF.vehicles.medium(v)),
      out: { green: 'long', priority: 'high', congestion: 'high' },
      text: 'IF density = High AND wait = Medium AND vehicles = Medium → green = Long' },
    { id: 6, cond: (v, d, w, e) => Math.min(MF.density.high(d), MF.waiting.short(w), MF.vehicles.high(v)),
      out: { green: 'long', priority: 'high', congestion: 'high' },
      text: 'IF density = High AND wait = Short AND vehicles = High → green = Long' },
    { id: 7, cond: (v, d, w, e) => Math.min(MF.density.high(d), MF.waiting.short(w), MF.vehicles.medium(v)),
      out: { green: 'medium', priority: 'high', congestion: 'high' },
      text: 'IF density = High AND wait = Short AND vehicles = Medium → green = Medium' },
    
    // Medium density
    { id: 8, cond: (v, d, w, e) => Math.min(MF.density.medium(d), MF.waiting.long(w), MF.vehicles.high(v)),
      out: { green: 'long', priority: 'high', congestion: 'high' },
      text: 'IF density = Medium AND wait = Long AND vehicles = High → green = Long' },
    { id: 9, cond: (v, d, w, e) => Math.min(MF.density.medium(d), MF.waiting.medium(w), MF.vehicles.medium(v)),
      out: { green: 'medium', priority: 'medium', congestion: 'moderate' },
      text: 'IF density = Medium AND wait = Medium AND vehicles = Medium → green = Medium' },
    { id: 10, cond: (v, d, w, e) => Math.min(MF.density.medium(d), MF.waiting.short(w), MF.vehicles.medium(v)),
      out: { green: 'short', priority: 'medium', congestion: 'moderate' },
      text: 'IF density = Medium AND wait = Short AND vehicles = Medium → green = Short' },
    { id: 11, cond: (v, d, w, e) => Math.min(MF.density.medium(d), MF.waiting.short(w), MF.vehicles.low(v)),
      out: { green: 'short', priority: 'low', congestion: 'low' },
      text: 'IF density = Medium AND wait = Short AND vehicles = Low → green = Short' },
    
    // Low density
    { id: 12, cond: (v, d, w, e) => Math.min(MF.density.low(d), MF.waiting.long(w), MF.vehicles.high(v)),
      out: { green: 'medium', priority: 'medium', congestion: 'moderate' },
      text: 'IF density = Low AND wait = Long AND vehicles = High → green = Medium' },
    { id: 13, cond: (v, d, w, e) => Math.min(MF.density.low(d), MF.waiting.medium(w), MF.vehicles.medium(v)),
      out: { green: 'short', priority: 'low', congestion: 'low' },
      text: 'IF density = Low AND wait = Medium AND vehicles = Medium → green = Short' },
    { id: 14, cond: (v, d, w, e) => Math.min(MF.density.low(d), MF.waiting.short(w), MF.vehicles.low(v)),
      out: { green: 'very_short', priority: 'low', congestion: 'low' },
      text: 'IF density = Low AND wait = Short AND vehicles = Low → green = Very Short' },
    { id: 15, cond: (v, d, w, e) => Math.min(MF.density.low(d), MF.waiting.short(w), MF.vehicles.medium(v)),
      out: { green: 'very_short', priority: 'low', congestion: 'low' },
      text: 'IF density = Low AND wait = Short AND vehicles = Medium → green = Very Short' },
];

/**
 * Run fuzzy inference on given inputs.
 * Uses weighted average defuzzification.
 */
function fuzzyInference(vehicles, density, waiting, emergency) {
    let greenNum = 0, greenDen = 0;
    let prioNum = 0, prioDen = 0;
    let congNum = 0, congDen = 0;
    const activeRules = [];

    for (const rule of FUZZY_RULES) {
        const strength = rule.cond(vehicles, density, waiting, emergency);
        if (strength > 0.001) {
            activeRules.push({ id: rule.id, strength: strength.toFixed(3) });
            greenNum += strength * OUTPUT.green[rule.out.green];
            greenDen += strength;
            prioNum += strength * OUTPUT.priority[rule.out.priority];
            prioDen += strength;
            congNum += strength * OUTPUT.congestion[rule.out.congestion];
            congDen += strength;
        }
    }

    const greenDuration = greenDen > 0 ? greenNum / greenDen : 30;
    const signalPriority = prioDen > 0 ? prioNum / prioDen : 2;
    const congestionLevel = congDen > 0 ? congNum / congDen : 30;

    return {
        green_duration: Math.round(Math.max(10, Math.min(120, greenDuration)) * 100) / 100,
        signal_priority: Math.round(Math.max(1, Math.min(5, signalPriority)) * 100) / 100,
        congestion_level: Math.round(Math.max(0, Math.min(100, congestionLevel)) * 100) / 100,
        active_rules: activeRules
    };
}

/**
 * Simulated neural network prediction (approximation).
 * In production, this would call the Python backend via API.
 */
function nnPrediction(vehicles, density, waiting, emergency) {
    // Learned approximation of the NN behavior
    const base = 15 + (vehicles / 100) * 45;
    const densBonus = (density / 100) * 25;
    const waitBonus = (waiting / 180) * 20;
    const emergBonus = 30 * emergency;
    
    const green = Math.max(10, Math.min(120, base + densBonus + waitBonus + emergBonus));
    
    const score = (vehicles / 100) * 0.3 + (density / 100) * 0.4 + (waiting / 180) * 0.3;
    let priority = emergency ? 5 : (score > 0.75 ? 4 : score > 0.55 ? 3 : score > 0.3 ? 2 : 1);
    
    const congestion = (0.35 * vehicles / 100 + 0.40 * density / 100 + 0.25 * waiting / 180) * 100;

    return {
        green_duration: Math.round(green * 100) / 100,
        signal_priority: priority,
        congestion_level: Math.round(Math.max(0, Math.min(100, congestion)) * 100) / 100
    };
}

/**
 * Neuro-Fuzzy combined prediction with adaptive weights.
 */
function neuroFuzzyPredict(vehicles, density, waiting, emergency) {
    const fuzzy = fuzzyInference(vehicles, density, waiting, emergency);
    const nn = nnPrediction(vehicles, density, waiting, emergency);
    
    // Adaptive weights
    let fw, nw;
    if (emergency === 1) {
        fw = 0.7; nw = 0.3;
    } else {
        const extremity = Math.max(vehicles / 100, density / 100, waiting / 180);
        if (extremity > 0.85) { fw = 0.5; nw = 0.5; }
        else if (extremity > 0.6) { fw = 0.4; nw = 0.6; }
        else { fw = 0.35; nw = 0.65; }
    }

    return {
        green_duration: Math.round((fw * fuzzy.green_duration + nw * nn.green_duration) * 100) / 100,
        signal_priority: Math.round((fw * fuzzy.signal_priority + nw * nn.signal_priority) * 100) / 100,
        congestion_level: Math.round((fw * fuzzy.congestion_level + nw * nn.congestion_level) * 100) / 100,
        fuzzy_weight: fw,
        nn_weight: nw,
        active_rules: fuzzy.active_rules
    };
}


// ============================================================
// UI UPDATE FUNCTIONS
// ============================================================

function updateInput() {
    const vehicles = parseInt(document.getElementById('input-vehicles').value);
    const density = parseInt(document.getElementById('input-density').value);
    const waiting = parseInt(document.getElementById('input-waiting').value);
    const emergency = document.getElementById('input-emergency').checked ? 1 : 0;

    // Update display values
    document.getElementById('vehicles-value').textContent = vehicles;
    document.getElementById('density-value').textContent = density + '%';
    document.getElementById('waiting-value').textContent = waiting + 's';
    document.getElementById('emergency-value').textContent = emergency ? 'YES 🚨' : 'No';

    // Run neuro-fuzzy inference
    const result = neuroFuzzyPredict(vehicles, density, waiting, emergency);

    // Update output rings
    updateRing('ring-green-fill', result.green_duration, 120);
    updateRing('ring-priority-fill', result.signal_priority, 5);
    updateRing('ring-congestion-fill', result.congestion_level, 100);

    document.getElementById('output-green').textContent = result.green_duration + 's';
    document.getElementById('output-priority').textContent = result.signal_priority.toFixed(1);
    document.getElementById('output-congestion').textContent = Math.round(result.congestion_level) + '%';

    // Update weights
    const fwPct = Math.round(result.fuzzy_weight * 100);
    const nwPct = Math.round(result.nn_weight * 100);
    document.getElementById('fuzzy-bar').style.width = fwPct + '%';
    document.getElementById('nn-bar').style.width = nwPct + '%';
    document.getElementById('fuzzy-weight').textContent = fwPct + '%';
    document.getElementById('nn-weight').textContent = nwPct + '%';

    // Update active rules
    updateActiveRules(result.active_rules);
}

function updateRing(id, value, max) {
    const circumference = 2 * Math.PI * 52; // r=52
    const offset = circumference - (value / max) * circumference;
    document.getElementById(id).style.strokeDashoffset = offset;
}

function updateActiveRules(activeRules) {
    const activeIds = new Set(activeRules.map(r => r.id));
    document.querySelectorAll('.rule-card').forEach(card => {
        const ruleId = parseInt(card.dataset.ruleId);
        card.classList.toggle('active', activeIds.has(ruleId));
    });
}

// Populate rules container
function populateRules() {
    const container = document.getElementById('rules-container');
    container.innerHTML = '';
    
    FUZZY_RULES.forEach(rule => {
        const card = document.createElement('div');
        card.className = 'rule-card';
        card.dataset.ruleId = rule.id;
        
        // Format rule text with colored keywords
        const formattedText = rule.text
            .replace(/IF/g, '<span class="rule-keyword">IF</span>')
            .replace(/AND/g, '<span class="rule-keyword">AND</span>')
            .replace(/→/g, '<span class="rule-keyword">→</span>')
            .replace(/(density|wait|vehicles|emergency|green|priority|congestion)/gi, '<span class="rule-variable">$1</span>')
            .replace(/(High|Medium|Low|Long|Short|Very Long|Very Short|Emergency|YES)/g, '<span class="rule-value">$1</span>');
        
        card.innerHTML = `
            <div class="rule-number">RULE R${rule.id}</div>
            <div class="rule-text">${formattedText}</div>
        `;
        container.appendChild(card);
    });
}


// ============================================================
// TRAFFIC SIMULATION
// ============================================================

let simulationRunning = false;
let simulationInterval = null;
let simSpeed = 5;
let emergencyMode = false;
let rushHourMode = false;

// Road states
const roads = {
    north: { vehicles: 0, density: 0, waiting: 0, greenTime: 0, state: 'red', timer: 0 },
    south: { vehicles: 0, density: 0, waiting: 0, greenTime: 0, state: 'red', timer: 0 },
    east:  { vehicles: 0, density: 0, waiting: 0, greenTime: 0, state: 'red', timer: 0 },
    west:  { vehicles: 0, density: 0, waiting: 0, greenTime: 0, state: 'red', timer: 0 }
};

let currentGreenPair = 'ns'; // 'ns' or 'ew'
let phaseTimer = 0;
let currentPhaseGreenTime = 45;

function toggleSimulation() {
    if (simulationRunning) {
        stopSimulation();
    } else {
        startSimulation();
    }
}

function startSimulation() {
    simulationRunning = true;
    document.getElementById('btn-start').innerHTML = '<span class="btn-icon">⏸</span> Pause Simulation';
    
    // Initialize
    generateTraffic();
    updateSignalPhase();
    
    simulationInterval = setInterval(() => {
        simulationTick();
    }, 1000 / simSpeed);
}

function stopSimulation() {
    simulationRunning = false;
    document.getElementById('btn-start').innerHTML = '<span class="btn-icon">▶</span> Start Simulation';
    if (simulationInterval) {
        clearInterval(simulationInterval);
        simulationInterval = null;
    }
}

function simulationTick() {
    phaseTimer++;
    
    // Generate random vehicle arrivals
    if (phaseTimer % 3 === 0) {
        generateTraffic();
    }
    
    // Green vehicles pass through (reduce count)
    if (currentGreenPair === 'ns') {
        roads.north.vehicles = Math.max(0, roads.north.vehicles - Math.floor(Math.random() * 3));
        roads.south.vehicles = Math.max(0, roads.south.vehicles - Math.floor(Math.random() * 3));
        roads.east.waiting += 2;
        roads.west.waiting += 2;
        roads.north.waiting = Math.max(0, roads.north.waiting - 5);
        roads.south.waiting = Math.max(0, roads.south.waiting - 5);
    } else {
        roads.east.vehicles = Math.max(0, roads.east.vehicles - Math.floor(Math.random() * 3));
        roads.west.vehicles = Math.max(0, roads.west.vehicles - Math.floor(Math.random() * 3));
        roads.north.waiting += 2;
        roads.south.waiting += 2;
        roads.east.waiting = Math.max(0, roads.east.waiting - 5);
        roads.west.waiting = Math.max(0, roads.west.waiting - 5);
    }
    
    // Clamp waiting times
    Object.values(roads).forEach(r => r.waiting = Math.min(180, Math.max(0, r.waiting)));
    
    // Check if phase should change
    if (phaseTimer >= currentPhaseGreenTime) {
        switchPhase();
    }
    
    // Update timers
    const remaining = currentPhaseGreenTime - phaseTimer;
    updateSimDisplay(remaining);
}

function generateTraffic() {
    const multiplier = rushHourMode ? 1.5 : 1;
    
    Object.keys(roads).forEach(dir => {
        const arrival = Math.floor(Math.random() * 8 * multiplier);
        roads[dir].vehicles = Math.min(100, roads[dir].vehicles + arrival);
        roads[dir].density = Math.min(100, roads[dir].vehicles + Math.random() * 10);
    });
}

function switchPhase() {
    phaseTimer = 0;
    currentGreenPair = currentGreenPair === 'ns' ? 'ew' : 'ns';
    
    // Use neuro-fuzzy to determine next green time
    const primaryRoads = currentGreenPair === 'ns' ? ['north', 'south'] : ['east', 'west'];
    const avgVehicles = (roads[primaryRoads[0]].vehicles + roads[primaryRoads[1]].vehicles) / 2;
    const avgDensity = (roads[primaryRoads[0]].density + roads[primaryRoads[1]].density) / 2;
    const avgWaiting = (roads[primaryRoads[0]].waiting + roads[primaryRoads[1]].waiting) / 2;
    
    const result = neuroFuzzyPredict(avgVehicles, avgDensity, avgWaiting, emergencyMode ? 1 : 0);
    currentPhaseGreenTime = Math.max(10, Math.min(60, Math.round(result.green_duration / 2)));
    
    updateSignalPhase();
}

function updateSignalPhase() {
    const directions = ['north', 'south', 'east', 'west'];
    
    directions.forEach(dir => {
        const isGreen = (currentGreenPair === 'ns' && (dir === 'north' || dir === 'south')) ||
                        (currentGreenPair === 'ew' && (dir === 'east' || dir === 'west'));
        
        document.getElementById(`${dir}-red`).classList.toggle('active', !isGreen);
        document.getElementById(`${dir}-yellow`).classList.remove('active');
        document.getElementById(`${dir}-green`).classList.toggle('active', isGreen);
    });
}

function updateSimDisplay(remaining) {
    const directions = ['north', 'south', 'east', 'west'];
    
    directions.forEach(dir => {
        document.getElementById(`${dir}-count`).textContent = roads[dir].vehicles + ' vehicles';
        
        const isGreen = (currentGreenPair === 'ns' && (dir === 'north' || dir === 'south')) ||
                        (currentGreenPair === 'ew' && (dir === 'east' || dir === 'west'));
        
        document.getElementById(`${dir}-timer`).textContent = isGreen ? remaining + 's' : '--';
    });
    
    // Update vehicle dots
    updateVehicleDots();
}

function updateVehicleDots() {
    const directions = ['north', 'south', 'east', 'west'];
    
    directions.forEach(dir => {
        const container = document.getElementById(`vehicles-${dir}`);
        if (!container) return;
        
        // Only update if count changed significantly
        const currentDots = container.children.length;
        const targetDots = Math.min(12, Math.floor(roads[dir].vehicles / 8));
        
        if (Math.abs(currentDots - targetDots) > 1) {
            container.innerHTML = '';
            for (let i = 0; i < targetDots; i++) {
                const dot = document.createElement('div');
                dot.className = `vehicle-dot ${emergencyMode && i === 0 ? 'emergency-car' : 'car'}`;
                
                // Position based on direction
                if (dir === 'north') {
                    dot.style.cssText = `position:absolute; left:${265 + (i % 3) * 14}px; top:${60 + Math.floor(i / 3) * 18}px;`;
                } else if (dir === 'south') {
                    dot.style.cssText = `position:absolute; left:${305 + (i % 3) * 14}px; bottom:${60 + Math.floor(i / 3) * 18}px;`;
                } else if (dir === 'east') {
                    dot.style.cssText = `position:absolute; right:${60 + Math.floor(i / 3) * 18}px; top:${265 + (i % 3) * 14}px;`;
                } else if (dir === 'west') {
                    dot.style.cssText = `position:absolute; left:${60 + Math.floor(i / 3) * 18}px; top:${305 + (i % 3) * 14}px;`;
                }
                
                container.appendChild(dot);
            }
        }
    });
}

function triggerEmergency() {
    emergencyMode = !emergencyMode;
    const btn = document.getElementById('btn-emergency');
    
    if (emergencyMode) {
        btn.innerHTML = '<span class="btn-icon">🚨</span> Cancel Emergency';
        btn.style.animation = 'pulse-glow 0.5s infinite';
        
        // Immediately switch phase and give max green time
        if (simulationRunning) {
            phaseTimer = 0;
            currentPhaseGreenTime = 60;
            updateSignalPhase();
        }
    } else {
        btn.innerHTML = '<span class="btn-icon">🚨</span> Emergency Vehicle';
        btn.style.animation = '';
    }
}

function triggerRushHour() {
    rushHourMode = !rushHourMode;
    const btn = document.getElementById('btn-rush');
    
    if (rushHourMode) {
        btn.innerHTML = '<span class="btn-icon">🕐</span> End Rush Hour';
        btn.style.background = 'linear-gradient(135deg, var(--accent-orange), var(--accent-yellow))';
        btn.style.color = '#000';
        btn.style.borderColor = 'transparent';
        
        // Flood with vehicles
        Object.values(roads).forEach(r => {
            r.vehicles = Math.min(100, r.vehicles + 30);
        });
    } else {
        btn.innerHTML = '<span class="btn-icon">🕐</span> Rush Hour';
        btn.style.background = '';
        btn.style.color = '';
        btn.style.borderColor = '';
    }
}

function updateSpeed(val) {
    simSpeed = parseInt(val);
    document.getElementById('speed-label').textContent = val + 'x';
    
    if (simulationRunning) {
        clearInterval(simulationInterval);
        simulationInterval = setInterval(() => simulationTick(), 1000 / simSpeed);
    }
}


// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    populateRules();
    updateInput();
    
    // Set initial signal states (North-South green)
    document.getElementById('north-red').classList.remove('active');
    document.getElementById('north-green').classList.add('active');
    document.getElementById('south-red').classList.remove('active');
    document.getElementById('south-green').classList.add('active');
    document.getElementById('east-red').classList.add('active');
    document.getElementById('west-red').classList.add('active');
    
    // Animate sections on scroll
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });
    
    document.querySelectorAll('.section').forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'all 0.6s ease-out';
        observer.observe(section);
    });
});

// ============================================================
// LIGHTBOX MODAL HANDLER
// ============================================================

function openLightbox(src, title) {
    const modal = document.getElementById('lightbox-modal');
    const img = document.getElementById('lightbox-img');
    const titleEl = document.getElementById('lightbox-title');
    
    if (modal && img && titleEl) {
        img.src = src;
        titleEl.textContent = title;
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
}

function closeLightbox(e) {
    if (e && e.target && e.target.closest('.lightbox-content') && !e.target.classList.contains('lightbox-close')) {
        return;
    }
    const modal = document.getElementById('lightbox-modal');
    if (modal) {
        modal.classList.remove('active');
        document.body.style.overflow = '';
    }
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeLightbox();
    }
});

