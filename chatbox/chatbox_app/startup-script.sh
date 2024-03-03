#! /bin/bash
docker start b308018212ec
gcloud compute instances add-tags {} --tags=session-{} --impersonate-service-account={} --zone={}