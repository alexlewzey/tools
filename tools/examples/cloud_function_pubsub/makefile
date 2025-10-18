create_topic:
	gcloud pubsub topics create cloud-function-topic

create_function:
	gcloud functions deploy hello-pubsub \
		--gen2 \
		--runtime python311 \
		--region us-central1 \
		--source . \
		--entry-point hello_pubsub \
		--trigger-topic cloud-function-topic

publish_message:
	gcloud pubsub topics publish cloud-function-topic --message '{"name":"Mole","age":11}'


logs:
	gcloud functions logs read hello-pubsub --limit 20
