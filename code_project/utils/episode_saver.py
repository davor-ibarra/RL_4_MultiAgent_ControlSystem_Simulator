import json
import os
import logging
from utils.numpy_encoder import NumpyEncoder # Ensure this handles numpy types correctly

# Configure logging for this module if run standalone, or rely on main's config
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_episode_batch(batch_data, results_folder, last_episode_number_in_batch):
    """Saves a batch of episode data to a JSON file, named with episode range."""
    if not batch_data:
        logging.warning("Attempted to save an empty episode batch.")
        return

    try:
        # Determine the episode range for the filename
        first_episode_in_batch = batch_data[0].get('episode', 'unknown')
        # Use the provided last episode number for the end of the range
        # Ensure numbers are integers if possible for clean filenames
        try:
            f_ep = int(first_episode_in_batch)
            l_ep = int(last_episode_number_in_batch)
            filename_range = f"{f_ep}_to_{l_ep}"
        except (ValueError, TypeError):
            filename_range = f"{first_episode_in_batch}_to_{last_episode_number_in_batch}"

        filename = os.path.join(results_folder, f'simulation_data_{filename_range}.json')

        # Save the data
        with open(filename, 'w') as f:
            # Use indent for readability, but can increase file size
            json.dump(batch_data, f, cls=NumpyEncoder, indent=2)
        logging.info(f"Saved batch episodes {filename_range} to {filename}")

    except IndexError:
        logging.error("Cannot determine filename range: batch_data is likely empty.")
    except TypeError as e:
        logging.error(f"Error serializing batch data to JSON: {e}. Check data types.")
    except OSError as e:
        logging.error(f"OS error saving episode batch to {filename}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error saving episode batch to {filename}: {e}")


def save_metadata(metadata_dict, results_folder):
    """Saves the simulation metadata dictionary to metadata.json."""
    filename = os.path.join(results_folder, 'metadata.json')
    try:
        with open(filename, 'w') as f:
            # Use indent for better readability of the metadata file
            json.dump(metadata_dict, f, cls=NumpyEncoder, indent=4)
        logging.info(f"Metadata saved successfully to {filename}")
    except TypeError as e:
        logging.error(f"Error serializing metadata to JSON: {e}. Check data types in metadata_dict.")
    except OSError as e:
        logging.error(f"OS error saving metadata to {filename}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error saving metadata to {filename}: {e}")