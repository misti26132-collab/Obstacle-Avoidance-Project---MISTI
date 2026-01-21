from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="P1NhFGBs85JMn1bboIpV"
)

result = client.run_workflow(
    workspace_name="hamzeh-alqaqa",
    workflow_id="find-pillars",
    images={
        "image": "YOUR_IMAGE.jpg"
    },
    use_cache=True # cache workflow definition for 15 minutes
)
