import pickle
import redis
import io
import logging

def download_candidate_pool_object(redis_client, bucket_name, object_name):
    key = f"{bucket_name}:{object_name}"
    try: 
        serialized_cp = redis_client.get(key)
        if serialized_cp is None:
            logging.info(f"No data found for key: {key}")
            return None
        return pickle.loads(serialized_cp)
    except redis.RedisError as e:
        logging.error(f"Redis error for key {key}: {str(e)}")
        return None
    except pickle.UnpicklingError as e:
        logging.error(f"Unpickling error for key {key}: {str(e)}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error for key {key}: {str(e)}")
        return None

def upload_candidate_pool_object(redis_client, cp_object, bucket_name, object_name):
    key = f"{bucket_name}:{object_name}"
    try:
        serialized_cp = pickle.dumps(cp_object)
        redis_client.set(key, serialized_cp)
        logging.info(f"Successfully stored object at key: {key}")
        return True
    except Exception as e:
        logging.error(f"Error storing object at key {key}: {str(e)}")
        return False

