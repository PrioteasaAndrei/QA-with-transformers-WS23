from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpRequest
import googleapiclient.discovery
from google.oauth2 import service_account
import json 
from chatbox_app.models import Session
import os
from chatbox_app.serializers import SessionSerializer
import time
from dotenv import load_dotenv
load_dotenv()

def home(request):
    return render(request, 'home.html')

def close_instance(request):
    creds = service_account.Credentials.from_service_account_info(json.loads(os.getenv('CLOUD_CONFIG')))
    compute = googleapiclient.discovery.build('compute', 'v1', credentials=creds)
    project = os.getenv('PROJECT')
    zone = os.getenv('ZONE')
    instance = os.getenv('INSTANCE')
    compute.instances().stop(project=project, zone=zone, instance=instance).execute()
    render(request, 'close_instance.html')


def room(request):
    creds = service_account.Credentials.from_service_account_info(json.loads(os.getenv('CLOUD_CONFIG')))
    compute = googleapiclient.discovery.build('compute', 'v1', credentials=creds)
    
    project = os.getenv('PROJECT')
    zone = os.getenv('ZONE')
    instance = os.getenv('INSTANCE')
    serializer = SessionSerializer(data = {})

    if(serializer.is_valid()):
        serializer.save()

    startup_script = open(
        os.path.join(os.path.dirname(__file__), "startup-script.sh")
    ).read()

    latest_session_id = SessionSerializer(Session.objects.latest("id"), many=False).data["id"]

    result = compute.instances().get(project=project, zone=zone,
                                 instance=instance).execute()
    fingerprint = result["metadata"]["fingerprint"]
    kind = result["metadata"]["kind"]
    body = {
    "items": [{
        "key": "startup-script",
        "value": startup_script.format(latest_session_id, json.loads(os.getenv('CLOUD_CONFIG'))["client_email"]),
    }],
    "kind": kind,
    "fingerprint": fingerprint
    }

    compute.instances().setMetadata(project=project, zone=zone, instance=instance, body=body).execute()
    response = compute.instances().start(project=project, zone=zone, instance=instance).execute()
    # Polling is required for both checking if instance has been launched and startup script has been returned
    # It is also the used method in google documentation
    for i in range(20):
        result = compute.zoneOperations().get(
            project=project,
            zone=zone,
            operation=response["id"]).execute()

        if result['status'] == 'DONE':
            if 'error' in result:
                raise Exception(result['error'])
            break
        time.sleep(1)
    
    for i in range(30):
        result = compute.instances().get(project=project, zone=zone, instance=instance).execute()
        if f"session-{latest_session_id}" in result["tags"]["items"]:
            return redirect(f"http://{os.getenv('INSTANCE_IP')}:8501") 
        time.sleep(1.5)

    return redirect("home")

# Create your views here.
