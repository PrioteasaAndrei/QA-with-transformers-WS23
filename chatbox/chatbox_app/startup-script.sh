#! /bin/bash
docker start b308018212ec
gcloud compute instances add-tags instance-20240207-234040 --tags=session-{} --impersonate-service-account={} --zone=northamerica-northeast1-a