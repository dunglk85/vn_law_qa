#!/bin/sh
set -e

agent="${A2A_AGENT:-legal-research}"

case "$agent" in
    legal-research)        port=8101 ;;
    citation-checker)      port=8102 ;;
    response-synthesizer)  port=8103 ;;
    *)
        echo "Unknown A2A_AGENT: $agent" >&2
        exit 1
        ;;
esac

module="app.agents.a2a_servers.$(echo "$agent" | tr '-' '_')_server"
exec uvicorn "${module}:app" --host 0.0.0.0 --port "$port"
