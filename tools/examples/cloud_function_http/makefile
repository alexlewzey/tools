deploy:
	@gcloud functions deploy function-http \
    --runtime python311 \
    --trigger-http \
    --entry-point hello_world


run:
	@curl -s -X POST "https://function-http-849921355156.us-central1.run.app" \
	-H "Authorization: bearer $(gcloud auth print-identity-token)" \
	-H "Content-Type: application/json" \
	-d '{ "name": "Developer" }' | jq '.'


local-deploy:
	uv run functions-framework --target hello_world --debug


local-run:
	@curl -X POST http://127.0.0.1:8080 -d '{"hello": "Mole"}' -H 'Content-Type: application/json'
