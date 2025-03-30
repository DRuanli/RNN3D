# src/RNN3D/pipeline/stage_05_visualization_web.py
from src.RNN3D.config.configuration import ConfigurationManager
from src.RNN3D.components.visualization import RNAVisualizer
from src.RNN3D.components.web_interface import RNAWebInterface
import logging
import os
import threading

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s]: %(message)s",
    handlers=[
        logging.FileHandler("logs/visualization_web.log"),
        logging.StreamHandler()
    ]
)


def run_visualization():
    """
    Run the visualization component

    Returns:
        dict: Results of the visualization process
    """
    try:
        # Create configuration
        config = ConfigurationManager()
        visualization_config = config.get_visualization_config()

        # Initialize visualizer
        visualizer = RNAVisualizer(config=visualization_config)

        # Run visualization
        results = visualizer.run()

        if results:
            logging.info("Visualization completed successfully")
            return results
        else:
            logging.error("Visualization failed")
            return {}

    except Exception as e:
        logging.error(f"Error in visualization: {e}")
        raise e


def run_web_interface():
    """
    Run the web interface component

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create configuration
        config = ConfigurationManager()
        web_interface_config = config.get_web_interface_config()
        model_config = config.get_model_config()

        # Initialize web interface
        web_interface = RNAWebInterface(config=web_interface_config, model_config=model_config)

        # Run web interface
        result = web_interface.run()

        if result:
            logging.info("Web interface setup completed successfully")
            return True
        else:
            logging.error("Web interface setup failed")
            return False

    except Exception as e:
        logging.error(f"Error in web interface: {e}")
        raise e


def run_server_in_thread(config):
    """
    Run the Flask server in a separate thread

    Args:
        config: Web interface configuration
    """
    from src.RNN3D.components.web_interface import RNAWebInterface

    # Initialize web interface
    web_interface = RNAWebInterface(config=config)

    # Set up the Flask app
    app = web_interface.setup_flask_app()

    # Run the server
    app.run(
        host=config.host,
        port=config.port,
        debug=False,  # Debug must be False for threading
        use_reloader=False  # Disable reloader for threading
    )


def main():
    """
    Main function for Stage 5
    """
    try:
        # Run visualization
        logging.info("Starting visualization process...")
        visualization_results = run_visualization()

        # Set up web interface
        logging.info("Setting up web interface...")
        web_interface_success = run_web_interface()

        if web_interface_success:
            # Get the configuration for running the server
            config = ConfigurationManager()
            web_interface_config = config.get_web_interface_config()

            # If run_server is enabled, start the server in a separate thread
            if web_interface_config.run_server:
                logging.info(f"Starting web server on port {web_interface_config.port}...")

                # Create and start the server thread
                server_thread = threading.Thread(
                    target=run_server_in_thread,
                    args=(web_interface_config,)
                )
                server_thread.daemon = True  # Allow the program to exit even if thread is running
                server_thread.start()

                logging.info(
                    f"Web server started in background thread. Access at http://{web_interface_config.host}:{web_interface_config.port}")
                logging.info("Press Ctrl+C to stop the server")

                # Keep the main thread alive
                try:
                    while True:
                        import time
                        time.sleep(1)
                except KeyboardInterrupt:
                    logging.info("Stopping server...")

        logging.info("Stage 5 completed successfully")
        return True

    except Exception as e:
        logging.error(f"Error in Stage 5: {e}")
        raise e


if __name__ == "__main__":
    main()