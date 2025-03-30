# src/RNN3D/components/web_interface.py
import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from src.RNN3D.entity.config_entity import WebInterfaceConfig
from src.RNN3D.components.vienna_model import ViennaRNAPredictor


class RNAWebInterface:
    def __init__(self, config=None, model_config=None):
        """
        Initialize the RNA Web Interface

        Args:
            config: Configuration for web interface (WebInterfaceConfig or similar)
            model_config: Optional configuration for the model (if prediction is enabled)
        """
        self.config = config
        self.model_config = model_config
        self.app = None

        # Use default values if config is None
        if self.config is None:
            logging.warning("Web interface configuration is None. Using default values.")
            from pathlib import Path

            # Create a simple configuration structure
            class DefaultConfig:
                def __init__(self):
                    self.templates_dir = Path("templates")
                    self.static_dir = Path("static")
                    self.visualizations_dir = Path("artifacts/visualization/visualizations")
                    self.metrics_path = Path("artifacts/model/validation/metrics.txt")
                    self.host = "0.0.0.0"
                    self.port = 5001
                    self.debug_mode = False
                    self.run_server = True
                    self.max_sequence_length = 480

            self.config = DefaultConfig()

    def setup_flask_app(self):
        """
        Sets up the Flask application for the web interface

        Returns:
            Flask: The configured Flask application
        """
        app = Flask(
            __name__,
            template_folder=str(Path(self.config.templates_dir).absolute()),
            static_folder=str(Path(self.config.static_dir).absolute())
        )

        # Register routes
        self._register_routes(app)

        self.app = app
        return app

    def _register_routes(self, app):
        """
        Registers all routes for the Flask application

        Args:
            app (Flask): The Flask application
        """

        # Home route
        @app.route('/')
        def home():
            return render_template('index.html')

        # Route to get visualization data
        @app.route('/api/visualization-data')
        def get_visualization_data():
            try:
                data_path = Path(self.config.visualizations_dir) / "visualization_data.json"
                if not data_path.exists():
                    return jsonify({'error': 'Visualization data not found'}), 404

                with open(data_path, 'r') as f:
                    data = json.load(f)

                return jsonify(data)
            except Exception as e:
                logging.error(f"Error loading visualization data: {e}")
                return jsonify({'error': str(e)}), 500

        # Route to serve static visualization images
        @app.route('/visualizations/<path:filename>')
        def get_visualization(filename):
            return send_from_directory(self.config.visualizations_dir, filename)

        # Route to predict structure for a new sequence
        @app.route('/api/predict', methods=['POST'])
        def predict_structure():
            try:
                # Get the sequence from the request
                data = request.get_json()
                sequence = data.get('sequence', '').strip().upper()

                if not sequence:
                    return jsonify({'error': 'No sequence provided'}), 400

                # Validate sequence
                valid_nucleotides = set('ACGU')
                if not all(n in valid_nucleotides for n in sequence):
                    return jsonify({'error': 'Invalid sequence. Only A, C, G, U allowed.'}), 400

                if len(sequence) > self.config.max_sequence_length:
                    return jsonify(
                        {'error': f'Sequence too long. Maximum length is {self.config.max_sequence_length}'}), 400

                # If model_config is provided, predict the structure
                if self.model_config:
                    # Initialize the model
                    model = ViennaRNAPredictor(config=self.model_config)

                    # Create a temporary dataframe for the sequence
                    df = pd.DataFrame({
                        'target_id': ['user_input'],
                        'sequence': [sequence]
                    })

                    # Predict the structure
                    solution = model.predict_structure(df)

                    # Extract the predicted coordinates
                    coords = solution['user_input']['coord'][0]  # First conformation

                    # Prepare the response
                    response = {
                        'sequence': sequence,
                        'secondary_structure': model.predict_secondary_structure(sequence),
                        'coordinates': coords.tolist(),
                        'nucleotides': list(sequence)
                    }

                    return jsonify(response)
                else:
                    return jsonify({'error': 'Prediction not enabled in this deployment'}), 501

            except Exception as e:
                logging.error(f"Error predicting structure: {e}")
                return jsonify({'error': str(e)}), 500

        # Route to get metrics from validation
        @app.route('/api/metrics')
        def get_metrics():
            try:
                metrics_path = Path(self.config.metrics_path)
                if not metrics_path.exists():
                    return jsonify({'error': 'Metrics data not found'}), 404

                metrics = {}
                with open(metrics_path, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            metrics[key.strip()] = value.strip()

                return jsonify(metrics)
            except Exception as e:
                logging.error(f"Error loading metrics: {e}")
                return jsonify({'error': str(e)}), 500

    def prepare_templates(self):
        """
        Prepares the templates for the web interface

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.config:
                logging.error("Web interface configuration is None")
                return False

            # Get templates directory with default fallback
            templates_dir = Path(getattr(self.config, 'templates_dir', 'templates'))
            os.makedirs(templates_dir, exist_ok=True)

            # Get static directory with default fallback
            static_dir = Path(getattr(self.config, 'static_dir', 'static'))
            os.makedirs(static_dir, exist_ok=True)
            os.makedirs(static_dir / "css", exist_ok=True)
            os.makedirs(static_dir / "js", exist_ok=True)

            # Create the index.html template
            index_html = templates_dir / "index.html"
            with open(index_html, 'w') as f:
                f.write(self._get_index_html_content())

            # Create the CSS file
            css_file = static_dir / "css" / "styles.css"
            with open(css_file, 'w') as f:
                f.write(self._get_css_content())

            # Create the JavaScript file
            js_file = static_dir / "js" / "main.js"
            with open(js_file, 'w') as f:
                f.write(self._get_js_content())

            logging.info(f"Prepared web interface templates in {templates_dir}")
            return True

        except Exception as e:
            logging.error(f"Error preparing templates: {e}")
            return False

    def _get_index_html_content(self):
        """
        Returns the content for the index.html template

        Returns:
            str: HTML content
        """
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RNN3D: RNA 3D Structure Prediction</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <header>
        <h1>RNN3D: RNA 3D Structure Prediction</h1>
        <p>Advanced RNA 3D Structure Prediction Using Machine Learning</p>
    </header>

    <main>
        <section class="dashboard">
            <div class="metrics-panel">
                <h2>Performance Metrics</h2>
                <div id="metrics-container">
                    <p>Loading metrics...</p>
                </div>
            </div>

            <div class="visualization-panel">
                <h2>RNA Structure Visualization</h2>

                <div class="controls">
                    <label for="rna-selector">Select RNA:</label>
                    <select id="rna-selector">
                        <option value="">Select RNA sequence</option>
                    </select>

                    <label for="conformation-selector">Conformation:</label>
                    <select id="conformation-selector">
                        <option value="">Select conformation</option>
                    </select>
                </div>

                <div id="visualization-container">
                    <div id="3d-container"></div>
                    <div id="sequence-display"></div>
                </div>
            </div>
        </section>

        <section class="prediction-form">
            <h2>Predict New RNA Structure</h2>

            <form id="prediction-form">
                <div class="form-group">
                    <label for="rna-sequence">RNA Sequence (A, C, G, U only):</label>
                    <textarea id="rna-sequence" placeholder="Enter RNA sequence (max 480 nucleotides)" maxlength="480" required></textarea>
                </div>

                <button type="submit" id="predict-button">Predict Structure</button>
            </form>

            <div id="prediction-results" style="display: none;">
                <h3>Prediction Results</h3>
                <div class="results-container">
                    <div class="secondary-structure">
                        <h4>Secondary Structure</h4>
                        <pre id="secondary-structure-display"></pre>
                    </div>

                    <div class="tertiary-structure">
                        <h4>3D Structure</h4>
                        <div id="prediction-3d-container"></div>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <footer>
        <p>&copy; 2025 RNN3D Project | Advanced RNA 3D Structure Prediction</p>
    </footer>

    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
"""

    def _get_css_content(self):
        """
        Returns the content for the CSS file

        Returns:
            str: CSS content
        """
        return """/* Global Styles */
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Arial', sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f4f7f9;
}

header {
    background-color: #2c3e50;
    color: white;
    text-align: center;
    padding: 2rem 0;
}

header h1 {
    margin-bottom: 0.5rem;
}

main {
    max-width: 1200px;
    margin: 2rem auto;
    padding: 0 2rem;
}

section {
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    padding: 2rem;
    margin-bottom: 2rem;
}

h2 {
    color: #2c3e50;
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #eaeaea;
}

/* Dashboard */
.dashboard {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 2rem;
}

.metrics-panel, .visualization-panel {
    overflow: hidden;
}

.metrics-panel {
    background-color: #f8f9fa;
    padding: 1.5rem;
    border-radius: 8px;
}

#metrics-container {
    display: grid;
    gap: 1rem;
}

.metric-card {
    background-color: white;
    padding: 1rem;
    border-radius: 6px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.metric-card h3 {
    font-size: 0.9rem;
    color: #6c757d;
    margin-bottom: 0.5rem;
}

.metric-card .value {
    font-size: 1.5rem;
    font-weight: bold;
    color: #2c3e50;
}

/* Visualization */
.controls {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    align-items: center;
}

.controls label {
    font-weight: bold;
    min-width: 100px;
}

.controls select {
    padding: 0.5rem;
    border-radius: 4px;
    border: 1px solid #ced4da;
    flex: 1;
}

#visualization-container {
    height: 500px;
    position: relative;
    border: 1px solid #e9ecef;
    border-radius: 8px;
    overflow: hidden;
}

#3d-container {
    height: 100%;
    width: 100%;
}

#sequence-display {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: rgba(255, 255, 255, 0.8);
    padding: 0.5rem;
    overflow-x: auto;
    white-space: nowrap;
    font-family: monospace;
}

/* Prediction Form */
.form-group {
    margin-bottom: 1.5rem;
}

.form-group label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: bold;
}

textarea {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid #ced4da;
    border-radius: 4px;
    min-height: 100px;
    font-family: monospace;
}

button {
    background-color: #3498db;
    color: white;
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 4px;
    cursor: pointer;
    font-weight: bold;
}

button:hover {
    background-color: #2980b9;
}

#prediction-results {
    margin-top: 2rem;
}

.results-container {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
    margin-top: 1rem;
}

.secondary-structure, .tertiary-structure {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 6px;
}

#secondary-structure-display {
    font-family: monospace;
    white-space: pre-wrap;
    margin-top: 1rem;
    line-height: 1.2;
}

#prediction-3d-container {
    height: 300px;
    border: 1px solid #e9ecef;
    border-radius: 4px;
    margin-top: 1rem;
}

/* Footer */
footer {
    text-align: center;
    padding: 2rem 0;
    background-color: #2c3e50;
    color: white;
}

/* Responsive adjustments */
@media screen and (max-width: 992px) {
    .dashboard {
        grid-template-columns: 1fr;
    }

    .results-container {
        grid-template-columns: 1fr;
    }
}
"""

    def _get_js_content(self):
        """
        Returns the content for the JavaScript file

        Returns:
            str: JavaScript content
        """
        return """// Global variables
let visualizationData = null;
let currentVisualization = null;
let predictionVisualization = null;

// DOM Elements
const rnaSelector = document.getElementById('rna-selector');
const conformationSelector = document.getElementById('conformation-selector');
const visualizationContainer = document.getElementById('3d-container');
const sequenceDisplay = document.getElementById('sequence-display');
const metricsContainer = document.getElementById('metrics-container');
const predictionForm = document.getElementById('prediction-form');
const rnaSequenceInput = document.getElementById('rna-sequence');
const predictionResults = document.getElementById('prediction-results');
const secondaryStructureDisplay = document.getElementById('secondary-structure-display');
const prediction3dContainer = document.getElementById('prediction-3d-container');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Load visualization data
    fetchVisualizationData();

    // Load metrics
    fetchMetrics();

    // Set up event listeners
    rnaSelector.addEventListener('change', handleRNASelection);
    conformationSelector.addEventListener('change', handleConformationSelection);
    predictionForm.addEventListener('submit', handlePredictionSubmit);
});

// Fetch visualization data from the API
async function fetchVisualizationData() {
    try {
        const response = await fetch('/api/visualization-data');
        if (!response.ok) {
            throw new Error('Failed to load visualization data');
        }

        visualizationData = await response.json();
        populateRNASelector();
    } catch (error) {
        console.error('Error fetching visualization data:', error);
        alert('Failed to load visualization data. Please try refreshing the page.');
    }
}

// Fetch metrics from the API
async function fetchMetrics() {
    try {
        const response = await fetch('/api/metrics');
        if (!response.ok) {
            throw new Error('Failed to load metrics data');
        }

        const metrics = await response.json();
        displayMetrics(metrics);
    } catch (error) {
        console.error('Error fetching metrics:', error);
        metricsContainer.innerHTML = '<p>Failed to load metrics. Please try refreshing the page.</p>';
    }
}

// Display metrics in the dashboard
function displayMetrics(metrics) {
    metricsContainer.innerHTML = '';

    // For each metric, create a card
    for (const [key, value] of Object.entries(metrics)) {
        const metricCard = document.createElement('div');
        metricCard.className = 'metric-card';

        // Format the key for display
        const formattedKey = key
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());

        metricCard.innerHTML = `
            <h3>${formattedKey}</h3>
            <div class="value">${value}</div>
        `;

        metricsContainer.appendChild(metricCard);
    }
}

// Populate the RNA selector dropdown
function populateRNASelector() {
    if (!visualizationData) return;

    rnaSelector.innerHTML = '<option value="">Select RNA sequence</option>';

    // Add an option for each RNA sequence
    for (const rnaId in visualizationData) {
        const option = document.createElement('option');
        option.value = rnaId;
        option.textContent = `${rnaId} (${visualizationData[rnaId].length} nt)`;
        rnaSelector.appendChild(option);
    }
}

// Handle RNA selection change
function handleRNASelection() {
    const rnaId = rnaSelector.value;

    if (!rnaId) {
        // Clear the conformation selector
        conformationSelector.innerHTML = '<option value="">Select conformation</option>';
        // Clear the visualization
        if (currentVisualization) {
            currentVisualization.clear();
            currentVisualization = null;
        }
        sequenceDisplay.textContent = '';
        return;
    }

    const rnaData = visualizationData[rnaId];

    // Populate the conformation selector
    conformationSelector.innerHTML = '<option value="">Select conformation</option>';

    for (let i = 0; i < rnaData.conformations.length; i++) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = `Conformation ${rnaData.conformations[i].conformation_id}`;
        conformationSelector.appendChild(option);
    }

    // Display the sequence
    sequenceDisplay.textContent = rnaData.sequence;

    // Select the first conformation by default
    if (rnaData.conformations.length > 0) {
        conformationSelector.value = 0;
        handleConformationSelection();
    }
}

// Handle conformation selection change
function handleConformationSelection() {
    const rnaId = rnaSelector.value;
    const conformationIndex = conformationSelector.value;

    if (!rnaId || conformationIndex === '') {
        // Clear the visualization
        if (currentVisualization) {
            currentVisualization.clear();
            currentVisualization = null;
        }
        return;
    }

    // Get the selected conformation data
    const rnaData = visualizationData[rnaId];
    const conformationData = rnaData.conformations[conformationIndex];

    // Display the 3D visualization
    displayVisualization(
        conformationData.coordinates,
        conformationData.nucleotides,
        visualizationContainer
    );
}

// Display the 3D visualization using Three.js
function displayVisualization(coordinates, nucleotides, container) {
    // Clear the container
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }

    // Create a new Three.js scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f0f0);

    // Create a camera
    const camera = new THREE.PerspectiveCamera(
        75, container.clientWidth / container.clientHeight, 0.1, 1000
    );
    camera.position.z = 50;

    // Create a renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Add orbit controls
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;

    // Add lights
    const ambientLight = new THREE.AmbientLight(0x404040, 1);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(0, 1, 0);
    scene.add(directionalLight);

    // Define nucleotide colors
    const nucleotideColors = {
        'A': 0xff0000, // Red
        'C': 0x0000ff, // Blue
        'G': 0x00ff00, // Green
        'U': 0xff8000  // Orange
    };

    // Create materials for each nucleotide type
    const materials = {};
    for (const [nucleotide, color] of Object.entries(nucleotideColors)) {
        materials[nucleotide] = new THREE.MeshLambertMaterial({ color });
    }

    // Create a default material for unknown nucleotides
    const defaultMaterial = new THREE.MeshLambertMaterial({ color: 0x808080 });

    // Create geometry for nucleotides
    const nucleotideGeometry = new THREE.SphereGeometry(1, 16, 16);

    // Add nucleotides to the scene
    const nucleotideObjects = [];

    for (let i = 0; i < coordinates.length; i++) {
        const [x, y, z] = coordinates[i];
        const nucleotide = nucleotides[i];

        const material = materials[nucleotide] || defaultMaterial;
        const sphere = new THREE.Mesh(nucleotideGeometry, material);
        sphere.position.set(x, y, z);

        scene.add(sphere);
        nucleotideObjects.push(sphere);
    }

    // Add backbone
    const backbonePoints = coordinates.map(coords => new THREE.Vector3(...coords));
    const backboneGeometry = new THREE.BufferGeometry().setFromPoints(backbonePoints);
    const backboneMaterial = new THREE.LineBasicMaterial({ color: 0x000000, linewidth: 2 });
    const backbone = new THREE.Line(backboneGeometry, backboneMaterial);
    scene.add(backbone);

    // Center the view
    const center = new THREE.Vector3();
    const box = new THREE.Box3().setFromPoints(backbonePoints);
    box.getCenter(center);

    // Adjust camera and controls
    controls.target.copy(center);
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    camera.position.copy(center);
    camera.position.z += maxDim * 2;
    camera.far = maxDim * 10;
    camera.updateProjectionMatrix();

    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }

    animate();

    // Add resize handling
    function handleResize() {
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    }

    window.addEventListener('resize', handleResize);

    // Save reference to the visualization context for cleanup
    return {
        scene,
        camera,
        renderer,
        controls,
        nucleotideObjects,
        backbone,
        clear: function() {
            window.removeEventListener('resize', handleResize);
            renderer.dispose();
            nucleotideGeometry.dispose();
            backboneGeometry.dispose();
            backboneMaterial.dispose();

            for (const material of Object.values(materials)) {
                material.dispose();
            }
            defaultMaterial.dispose();

            container.removeChild(renderer.domElement);
        }
    };
}

// Handle form submission for predicting new structures
async function handlePredictionSubmit(event) {
    event.preventDefault();

    const sequence = rnaSequenceInput.value.trim().toUpperCase();

    // Validate the sequence
    if (!/^[ACGU]+$/.test(sequence)) {
        alert('Invalid sequence. Please enter only A, C, G, U nucleotides.');
        return;
    }

    if (sequence.length > 480) {
        alert('Sequence too long. Maximum length is 480 nucleotides.');
        return;
    }

    // Show loading indicator
    const submitButton = document.getElementById('predict-button');
    const originalButtonText = submitButton.textContent;
    submitButton.textContent = 'Predicting...';
    submitButton.disabled = true;

    try {
        // Send the prediction request
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ sequence })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Failed to predict structure');
        }

        const result = await response.json();

        // Display the results
        predictionResults.style.display = 'block';

        // Display the secondary structure
        secondaryStructureDisplay.textContent = `Sequence:\n${result.sequence}\n\nStructure:\n${result.secondary_structure}`;

        // Display the 3D structure
        const coordinates = result.coordinates;
        const nucleotides = result.nucleotides;

        // Clear any existing visualization
        if (predictionVisualization) {
            predictionVisualization.clear();
        }

        // Create a new visualization
        predictionVisualization = displayVisualization(
            coordinates,
            nucleotides,
            prediction3dContainer
        );

        // Scroll to the results
        predictionResults.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error('Error predicting structure:', error);
        alert(error.message || 'Failed to predict structure. Please try again.');
    } finally {
        // Reset the button
        submitButton.textContent = originalButtonText;
        submitButton.disabled = false;
    }
}
"""

    def run(self):
        """
        Runs the web interface setup

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Prepare the templates and static files
            if not self.prepare_templates():
                return False

            # Set up the Flask app
            app = self.setup_flask_app()

            logging.info(f"Web interface prepared successfully. Run with 'flask run' or app.run()")

            # If run_server is enabled, start the server
            if self.config.run_server:
                logging.info(f"Starting web server on port {self.config.port}...")
                app.run(
                    host=self.config.host,
                    port=self.config.port,
                    debug=self.config.debug_mode
                )

            return True

        except Exception as e:
            logging.error(f"Error setting up web interface: {e}")
            return False