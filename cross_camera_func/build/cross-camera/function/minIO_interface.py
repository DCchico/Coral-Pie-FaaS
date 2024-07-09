import pickle
from minio import Minio
from minio.error import S3Error
import io


# Serialize and upload the VehicleTracking object
def upload_candidate_pool_object(minio_client, cp_object, bucket_name, object_name):
    serialized_cp = pickle.dumps(cp_object)
    try:
        minio_client.put_object(
            bucket_name, object_name, io.BytesIO(serialized_cp), len(serialized_cp)
        )
    except Exception as e:
        print("An error occurred:", e)



# To retrieve the object
def download_candidate_pool_object(minio_client, bucket_name, object_name):
    try: 
        object_data = minio_client.get_object(bucket_name, object_name)
        serialized_vt = object_data.data
        return pickle.loads(serialized_vt)
    except S3Error as e:
        if e.code == "NoSuchKey":
            print("Object does not exist in MinIO.")
            return None  # Object not found
        else:
            print("An error occurred:", e)
            return None

