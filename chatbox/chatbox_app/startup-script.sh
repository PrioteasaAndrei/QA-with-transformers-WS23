#! /bin/bash
docker start bcb5f2076a3c
gcloud compute instances add-tags {} --tags=session-{} --impersonate-service-account={} --zone={}