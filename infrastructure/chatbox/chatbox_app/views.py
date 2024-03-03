from django.shortcuts import render, redirect
import googleapiclient.discovery
from google.oauth2 import service_account
import json 
from chatbox_app.models import Session
import os
from chatbox_app.serializers import SessionSerializer
import time
from dotenv import load_dotenv
from google.cloud import compute_v1
from django.views.decorators.cache import never_cache

@never_cache
def home(request):
    return render(request, 'home.html')

@never_cache
def close_instance(request):
    load_dotenv()
    creds = service_account.Credentials.from_service_account_info(json.loads(os.getenv('CLOUD_CONFIG')))
    compute = googleapiclient.discovery.build('compute', 'v1', credentials=creds)
    project, zone, instance = os.getenv('PROJECT'), os.getenv('ZONE'), os.getenv('INSTANCE')
    compute.instances().stop(project=project, zone=zone, instance=instance).execute()
    return render(request, 'close_instance.html')

def room(request):
    load_dotenv()
    creds = service_account.Credentials.from_service_account_info(json.loads(os.getenv('CLOUD_CONFIG')))
    compute = googleapiclient.discovery.build('compute', 'v1', credentials=creds)
    
    project, zone, instance = os.getenv('PROJECT'), os.getenv('ZONE'), os.getenv('INSTANCE')

    result = compute.instances().get(project=project, zone=zone, instance=instance).execute()
    latest_session_id = SessionSerializer(Session.objects.latest("id"), many=False).data["id"]
    if result["status"] == "RUNNING" and f"session-{latest_session_id}" in result["tags"]["items"]:
        return redirect(f"http://{os.getenv('INSTANCE_IP')}")
    
    serializer = SessionSerializer(data = {})    
    if(serializer.is_valid()):
        serializer.save()

    startup_script = open(os.path.join(os.path.dirname(__file__), "startup-script.sh")).read()
    latest_session_id = SessionSerializer(Session.objects.latest("id"), many=False).data["id"]

    fingerprint = result["metadata"]["fingerprint"]
    kind = result["metadata"]["kind"]
    body = {
    "items": [{
        "key": "startup-script",
        "value": startup_script.format(instance, latest_session_id, json.loads(os.getenv('CLOUD_CONFIG'))["client_email"], zone),
    }],
    "kind": kind,
    "fingerprint": fingerprint
    }

    compute.instances().setMetadata(project=project, zone=zone, instance=instance, body=body).execute()
    response = compute.instances().start(project=project, zone=zone, instance=instance).execute()
    # Following lines only waits instance to be start 
    kwargs = {"project": project, "operation": response["name"], "zone": response["zone"].rsplit("/", maxsplit=1)[1]}
    client = compute_v1.ZoneOperationsClient(credentials=creds)
    client.wait(**kwargs)
    # Polling is required for checking if startup script has been returned. It is also the used method in google documentation
    
    for i in range(40):
        print("try")
        result = compute.instances().get(project=project, zone=zone, instance=instance).execute()
        if f"session-{latest_session_id}" in result["tags"]["items"]:
            return redirect(f"http://{os.getenv('INSTANCE_IP')}") 
        time.sleep(1.5)

    return redirect("home")

# Create your views here.
