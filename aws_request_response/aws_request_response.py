from typing import Dict, Iterable, Union

import numpy as np
import joblib
import json
import pickle
import io
import uuid
import time

from botocore.exceptions import ClientError


def create_taskid() -> str:
    return uuid.uuid4().hex


def create_data_s3_key(prefix, taskid, name, file_format):
    key = f"{prefix}/{taskid}/{name}.{file_format}"
    return key

class AWSDirectData():
    cls_str = "DirectData"

    def __init__(self, name, value):
        self.value = value
        self.name = name

    @classmethod
    def from_message(cls, msg, s3=None):
        name = msg["Name"]
        value = msg["Value"]
        obj = AWSDirectData(name, value)
        return obj

    def to_message(self):
        return {"Type": self.cls_str, "Name": self.name, "Value": self.value}

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def get_name(self):
        return self.name

    def store_value(self):
        pass

    def delete_value(self):
        pass


class AWSS3Data():
    cls_str = "S3Data"

    def __init__(self, name:str, s3_bucket:str, s3_key:str, s3, value=None):
        self.s3 = s3
        self.s3_key = s3_key
        self.s3_bucket = s3_bucket
        self.name = name
        self.value = value

    @classmethod
    def from_message(cls, msg, s3):
        name = msg["Name"]
        bucket = msg["Bucket"]
        key = msg["Key"]
        obj = AWSS3Data(name, bucket, key, s3)
        return obj

    def to_message(self):
        return {"Type": self.cls_str, "Name": self.name, "Bucket": self.s3_bucket, "Key": self.s3_key}

    def get_name(self):
        return self.name

    def set_value(self, value):
        self.value = value

    def get_value(self):
        if self.value is None:
            self.value = self._fetch_value()
        return self.value

    def _fetch_value(self):
        obj = self.s3.get_object(Bucket=self.s3_bucket, Key=self.s3_key)
        body = obj["Body"]
        if self.s3_key.endswith(".json"):
            value = json.load(body)
        elif self.s3_key.endswith(".joblib"):
            body = io.BytesIO(body.read())
            value = joblib.load(body)
        elif self.s3_key.endswith(".pkl"):
            body = io.BytesIO(body.read())
            value = pickle.load(body)
        elif self.s3_key.endswith(".npy"):
            body = io.BytesIO(body.read())
            value = np.load(body)
        else:
            raise(Exception(f"No reader for key {self.s3_key}"))

        return value

    def store_value(self):
        body = io.BytesIO()
        if self.s3_key.endswith(".json"):
            body = json.dumps(self.value)
        elif self.s3_key.endswith(".joblib"):
            joblib.dump(self.value, body)
        elif self.s3_key.endswith(".pkl"):
            pickle.dump(self.value, body)
        elif self.s3_key.endswith(".npy"):
            np.save(body, self.value)
        else:
            raise(Exception(f"No writer for key {self.s3_key}"))
        if isinstance(body, io.BytesIO):
            body.seek(0)
        self.s3.put_object(Bucket=self.s3_bucket, Key=self.s3_key, Body=body)

    def delete_value(self):
        self.s3.delete_object(Bucket=self.s3_bucket, Key=self.s3_key)


Parameter = Union[AWSDirectData, AWSS3Data]


def parse_parameter(desc: dict, s3) -> Parameter:
    type = desc["Type"]
    if type == AWSS3Data.cls_str:
        param = AWSS3Data.from_message(desc, s3)
    elif type == AWSDirectData.cls_str:
        param = AWSDirectData.from_message(desc, s3)
    else:
        raise(Exception(f"Unknown parameter type {type}"))

    return param


def create_parameter(name: str, value, taskid: str, bucket: str, prefix: str, s3) -> Parameter:
    param = None
    if isinstance(value, (int, float, bool, str)):
        param = AWSDirectData(name, value)
    elif isinstance(value, np.ndarray):
        key = create_data_s3_key(prefix, taskid, name, "npy")
        param = AWSS3Data(name, bucket, key, s3, value)
    else:
        try:
            data = json.dumps(value)
            if len(data) < 1024:
                param = AWSDirectData(name, value)
        except: pass

    if param is None:
        key = create_data_s3_key(prefix, taskid, name, "joblib")
        param = AWSS3Data(name, bucket, key, s3, value)

    return param


class AWSTask:
    def __init__(self, taskid: str, parameters: Iterable[Parameter], result_data: AWSS3Data, s3, metadata=None):
        self.metadata = metadata
        self.parameters = parameters
        self.result_data = result_data
        self.s3 = s3
        self.taskid = taskid

    @classmethod
    def create_task(cls, bucket: str, prefix: str, s3, params: dict):
        """
        Creates a new task object. Parameters and results are communicated via
        S3 with path {bucket}/{prefix}/{taskid}/{object_name}, taskid is auto-generated
        and object_name is either the name of the parameter or 'results'.

        :param bucket: Identifier of S3 bucket to store parameters and results
        :param prefix: Prefix of stored objects inside S3
        :param s3: Boto3 S3 client
        :param params: parameter dictionary
        :return:
        """
        taskid = create_taskid()
        result_key = create_data_s3_key(prefix, taskid, "result", "joblib")
        result_data = AWSS3Data("result", bucket, result_key, s3)
        parameters = []
        for name, value in params.items():
            param = create_parameter(name, value, taskid, bucket, prefix, s3)
            parameters.append(param)
        task = AWSTask(taskid, parameters, result_data, s3)
        return task

    @classmethod
    def from_message(cls, message: dict, s3, metadata=None):
        taskid = message["TaskId"]
        parameters = []
        for p in message["Parameters"]:
            parameter = parse_parameter(p, s3=s3)
            parameters.append(parameter)
        result_data = AWSS3Data.from_message(message["ResultData"], s3)
        if metadata is None and "Metadata" in message:
            metadata = message["Metadata"]
        task = AWSTask(taskid=taskid, parameters=parameters, result_data=result_data, s3=s3, metadata=metadata)
        return task

    def to_message(self) -> dict:
        parameters = []
        for p in self.parameters:
            parameters.append(p.to_message())
        obj = {"TaskId": self.taskid,
               "ResultData": self.result_data.to_message(),
               "Parameters": parameters
               }
        if self.metadata is not None:
            obj["Metadata"] = self.metadata
        return obj

    def get_parameters(self) -> dict:
        values = {p.get_name(): p.get_value() for p in self.parameters}
        return values

    def store_result(self, result):
        self.result_data.set_value(result)
        self.result_data.store_value()

    def get_result(self):
        return self.result_data.get_value()


class AWSTaskServer:
    def __init__(self, task_queue: str, s3, sqs):
        self.sqs = sqs
        self.s3 = s3
        self.task_queue = task_queue
        self.exit_asap = False

    def fetch_task(self, wait_time_seconds=20, visibility_timeout=60*60) -> AWSTask:
        response = self.sqs.receive_message(
            QueueUrl=self.task_queue,
            VisibilityTimeout=visibility_timeout,
            WaitTimeSeconds=wait_time_seconds
        )
        if 'Messages' not in response:
            return None

        sqs_message = response['Messages'][0]
        message_body = json.loads(sqs_message["Body"])
        metadata = dict((k, sqs_message[k]) for k in ("MessageId", "ReceiptHandle"))
        task = AWSTask.from_message(message=message_body, s3=self.s3, metadata=metadata)
        return task

    def run(self, task_callback, max_tasks=-1, wait_time_seconds=20, visibility_timeout=60*60):
        task_count = 0
        while (max_tasks <= 0 or task_count < max_tasks) and not self.exit_asap:
            task = self.fetch_task(wait_time_seconds, visibility_timeout)
            if task is not None:
                params = task.get_parameters()
                result = task_callback(**params)
                self.complete_task(task, result)
                task_count += 1
                #TODO: Error handling

    def complete_task(self, task: AWSTask, result):
        task.store_result(result)
        self.sqs.delete_message(
            QueueUrl=self.task_queue,
            ReceiptHandle=task.metadata['ReceiptHandle'],
        )


class AWSResultFuture:
    def __init__(self, task: AWSTask):
        self.task = task

    def get_result(self):
        """
        Polls S3 whether result is available

        :return: None if result is not yet available, otherwise the result
        """
        try:
            value = self.task.result_data.get_value()
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                value = None
            else:
                raise(e)
        return value

    def cleanup(self):
        for p in self.task.parameters:
            p.delete_value()
        self.task.result_data.delete_value()


class AWSTaskClient:
    def __init__(self, task_queue: str, s3, sqs):
        self.sqs = sqs
        self.s3 = s3
        self.task_queue = task_queue

    def submit(self, task: AWSTask) -> AWSResultFuture:
        """
        Submits a job without blocking.

        :param task: The task object describing the data to be submitted
        :return: A future that allows to query for the result of the task
        """
        task.result_data.set_value(None)
        message_body = json.dumps(task.to_message())
        for param in task.parameters:
            param.store_value()
        response = self.sqs.send_message(QueueUrl=self.task_queue, MessageBody=message_body)
        future = AWSResultFuture(task)
        return future

    def submit_and_wait(self, task: AWSTask, timeout: int, poll_wait_time=10):
        """
        Submits a job and blocks until the result is available or until a specified timeout.

        :param task: The task object describing the data to be submitted
        :param timeout: The timeout (seconds). If there is no result available until then, a TimeoutError will be raised
        :param poll_wait_time: Time delay between two polls to S3
        :return: The result if available
        """
        future = self.submit(task)
        t_start = time.time()
        while time.time() - t_start < timeout:
            result = future.get_result()
            if result is not None:
                future.cleanup()
                return result
            time.sleep(poll_wait_time)
        raise(TimeoutError())
