# Data quality Monitor
from sagemaker.model_monitor import DefaultModelMonitor
from sagemaker.model_monitor.dataset_format import DatasetFormat
import boto3
import time
import botocore

from sagemaker import Session
import pandas as pd
import numpy as np

# Network Configuration
from sagemaker.network import NetworkConfig

from sagemaker.model import Model

from sagemaker.model_monitor import (
    CronExpressionGenerator,
)


class DemoDataQualityModelMonitor:

    def __init__(self, endpoint_name, bucket, projectfolder_prefix, training_dataset_path, kms_key,
                 record_preprocessor_script, post_analytics_processor_script,
                 subnets: list, security_group_ids: list, role, tags):
        self.endpoint_name = endpoint_name
        self.bucket = bucket
        self.prefix = projectfolder_prefix
        self.training_dataset_path = training_dataset_path
        self.record_preprocessor_script = record_preprocessor_script
        self.post_analytics_processor_script = post_analytics_processor_script
        self.kms_kwargs = dict(output_kms_key=kms_key, volume_kms_key=kms_key)
        self.network_config = NetworkConfig(subnets=subnets, security_group_ids=security_group_ids,
                                            enable_network_isolation=False, encrypt_inter_container_traffic=True)
        self.role = role
        self.sm_client = boto3.client('sagemaker')
        self.region = boto3.Session().region_name,
        self.tags = tags
        
    def create_data_quality_monitor(self):

        reports_prefix = '{}/data_quality/reports'.format(self.prefix)
        self.data_quality_s3_report_path = 's3://{}/{}'.format(self.bucket, reports_prefix)

        data_quality_prefix = self.prefix + '/data_quality'
        data_quality_baseline_prefix = data_quality_prefix + '/baselining'
        data_quality_baseline_results_prefix = data_quality_baseline_prefix + '/results'

        self.data_quality_baseline_results_uri = 's3://{}/{}'.format(self.bucket, data_quality_baseline_results_prefix)

        print('Baseline results uri: {}'.format(self.data_quality_baseline_results_uri))

        my_default_monitor = DefaultModelMonitor(
            role=self.role,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            volume_size_in_gb=20,
            max_runtime_in_seconds=3600,
            network_config=self.network_config,
            tags=self.tags,
            **self.kms_kwargs
        )

        my_default_monitor.suggest_baseline(
            baseline_dataset=self.training_dataset_path,
            dataset_format=DatasetFormat.csv(header=True),
            output_s3_uri=self.data_quality_baseline_results_uri,
            wait=True
        )

        my_default_monitor._validate_network_config = lambda network_config_dict: None


        mon_schedule_name = self.endpoint_name[:60] + '-dq'

        try:
            self.sm_client.delete_monitoring_schedule(MonitoringScheduleName=mon_schedule_name)
        except Exception as e:
            pass

        my_default_monitor.create_monitoring_schedule(
            monitor_schedule_name=mon_schedule_name,
            endpoint_input=self.endpoint_name,
            output_s3_uri=self.data_quality_s3_report_path,
            statistics=my_default_monitor.baseline_statistics(),
            constraints=my_default_monitor.suggested_constraints(),
            schedule_cron_expression=CronExpressionGenerator.hourly(),
            record_preprocessor_script=self.record_preprocessor_script,
            post_analytics_processor_script=self.post_analytics_processor_script,
            enable_cloudwatch_metrics=True,
        )

        print(mon_schedule_name)
        
        return my_default_monitor