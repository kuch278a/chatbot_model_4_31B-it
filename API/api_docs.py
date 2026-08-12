"""
API documentation blueprint.

Serves an OpenAPI 3.0 specification and a Swagger UI page
so the API can be explored and tested interactively.
"""

import json
from flask import Blueprint, Response

docs_bp = Blueprint("docs", __name__)

OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title": "Amani AI API",
        "description": (
            "Amani AI — Amharic-first multimodal conversational assistant. "
            "Provides chat (blocking + streaming), speech-to-text, "
            "text-to-speech, and RAG document management endpoints."
        ),
        "version": "1.0.0",
    },
    "servers": [
        {"url": "/", "description": "Same-origin (default host)"}
    ],
    "paths": {
        "/health": {
            "get": {
                "summary": "Service health check",
                "operationId": "getHealth",
                "responses": {
                    "200": {
                        "description": "All services ready",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string", "example": "ready"}
                                    }
                                }
                            }
                        }
                    },
                    "503": {
                        "description": "Services still loading",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string", "example": "loading"},
                                        "attempts": {"type": "integer"},
                                        "max_attempts": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/chat": {
            "post": {
                "summary": "Generate a chatbot response (blocking)",
                "description": (
                    "Returns the full response after generation completes. "
                    "Serialized by a global lock — only one LLM generation runs at a time."
                ),
                "operationId": "chat",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["prompt"],
                                "properties": {
                                    "prompt": {"type": "string", "example": "አማኒ ማነው?"},
                                    "session_id": {
                                        "type": "string",
                                        "default": "default_session",
                                        "example": "user-42"
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Generated response with RAG sources",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "response": {"type": "string"},
                                        "sources": {"type": "array", "items": {"type": "object"}}
                                    }
                                }
                            }
                        }
                    },
                    "400": {"description": "Empty prompt"},
                    "503": {"description": "Services not ready or server busy"},
                    "500": {"description": "Generation error"}
                }
            }
        },
        "/chat/stream": {
            "post": {
                "summary": "Stream a chatbot response token-by-token",
                "description": (
                    "Returns a text/plain stream (SSE-friendly, not buffered by nginx). "
                    "Tokens are yielded as the model generates them."
                ),
                "operationId": "chatStream",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["prompt"],
                                "properties": {
                                    "prompt": {"type": "string", "example": "ስለ አማኒ ተናገር"},
                                    "session_id": {
                                        "type": "string",
                                        "default": "default_session"
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Token stream (text/plain)",
                        "content": {
                            "text/plain": {
                                "schema": {"type": "string", "example": "አማኒ ..."}
                            }
                        }
                    },
                    "400": {"description": "Empty prompt"},
                    "503": {"description": "Services not ready or server busy"}
                }
            }
        },
        "/api/stt": {
            "post": {
                "summary": "Transcribe audio to text (speech-to-text)",
                "description": (
                    "Raw binary audio body from the browser MediaRecorder (WebM/opus). "
                    "Uses faster-whisper distil-large-v3.5 (int8, CPU)."
                ),
                "operationId": "stt",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/octet-stream": {
                            "schema": {"type": "string", "format": "binary"}
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Transcription result",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string", "example": "success"},
                                        "transcript": {"type": "string", "example": "ሰላም"}
                                    }
                                }
                            }
                        }
                    },
                    "400": {"description": "No audio data"},
                    "500": {"description": "Transcription error"}
                }
            }
        },
        "/api/voices": {
            "get": {
                "summary": "List available TTS voices",
                "operationId": "listVoices",
                "parameters": [
                    {
                        "name": "lang",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "string", "example": "am-ET"},
                        "description": "Filter voices by language (e.g. am-ET)"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "Voice list",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string", "example": "success"},
                                        "voices": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "short_name": {"type": "string"},
                                                    "lang": {"type": "string"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "/api/tts": {
            "post": {
                "summary": "Synthesize speech parameters (no audio)",
                "description": (
                    "Returns voice/rate metadata for the given text. "
                    "Use /api/tts/audio to get the actual MP3."
                ),
                "operationId": "tts",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["text"],
                                "properties": {
                                    "text": {"type": "string", "example": "እንኳን ወደ አማኒ ረዳት በደህና መጡ"},
                                    "lang": {"type": "string", "default": "am-ET"},
                                    "rate": {"type": "number", "format": "float", "default": 1.0},
                                    "pitch": {"type": "number", "format": "float", "default": 1.0}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Speech metadata",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string", "example": "success"},
                                        "text": {"type": "string"},
                                        "voice": {"type": "string"},
                                        "short_name": {"type": "string"},
                                        "lang": {"type": "string"},
                                        "rate": {"type": "number"},
                                        "pitch": {"type": "number"},
                                        "wpm": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    },
                    "400": {"description": "Empty text"},
                    "500": {"description": "Synthesis error"}
                }
            }
        },
        "/api/tts/audio": {
            "get": {
                "summary": "Generate and stream TTS audio (GET)",
                "description": "Returns an MP3 audio file synthesized with edge-tts.",
                "operationId": "ttsAudioGet",
                "parameters": [
                    {
                        "name": "text",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "string", "example": "እንኳን ወደ አማኒ ረዳት በደህና መጡ"}
                    },
                    {
                        "name": "lang",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "string", "default": "am-ET"}
                    }
                ],
                "responses": {
                    "200": {
                        "description": "MP3 audio stream",
                        "content": {
                            "audio/mpeg": {"schema": {"type": "string", "format": "binary"}}
                        }
                    },
                    "400": {"description": "Empty text"},
                    "500": {"description": "Synthesis error"}
                }
            },
            "post": {
                "summary": "Generate and stream TTS audio (POST)",
                "operationId": "ttsAudioPost",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "text": {"type": "string", "default": "እንኳን ወደ አማኒ ረዳት በደህና መጡ"},
                                    "lang": {"type": "string", "default": "am-ET"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "MP3 audio stream",
                        "content": {
                            "audio/mpeg": {"schema": {"type": "string", "format": "binary"}}
                        }
                    },
                    "400": {"description": "Empty text"},
                    "500": {"description": "Synthesis error"}
                }
            }
        },
        "/refresh": {
            "post": {
                "summary": "Rescan and re-index the documents folder",
                "operationId": "refresh",
                "responses": {
                    "200": {
                        "description": "Index refreshed",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string", "example": "success"},
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "503": {"description": "RAG pipeline not loaded"},
                    "500": {"description": "Indexing error"}
                }
            }
        },
        "/files": {
            "get": {
                "summary": "List all indexed document files",
                "operationId": "getFiles",
                "responses": {
                    "200": {
                        "description": "Indexed file names",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "files": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            }
                        }
                    },
                    "503": {"description": "Vector store not loaded"},
                    "500": {"description": "Query error"}
                }
            }
        },
        "/clear": {
            "post": {
                "summary": "Clear chat history for a session",
                "operationId": "clearHistory",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "session_id": {
                                        "type": "string",
                                        "default": "default_session"
                                    }
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "History cleared",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "status": {"type": "string", "example": "success"},
                                        "message": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "500": {"description": "Clear error"}
                }
            }
        }
    }
}

SWAGGER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Amani AI API Docs</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
  <style>
    html { box-sizing: border-box; overflow-y: scroll; }
    *, *:before, *:after { box-sizing: inherit; }
    .topbar { display: none; }
    body { margin: 0; background: #fafafa; }
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = function () {
      window.ui = SwaggerUIBundle({
        url: "/API/openapi.json",
        dom_id: "#swagger-ui",
        deepLinking: true,
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIBundle.SwaggerUIStandalonePreset
        ],
        layout: "BaseLayout"
      });
    };
  </script>
</body>
</html>
"""


@docs_bp.route("/openapi.json")
def openapi_json():
    """Serves the OpenAPI 3.0 specification."""
    return Response(
        json.dumps(OPENAPI_SPEC, indent=2, ensure_ascii=False),
        mimetype="application/json",
    )


@docs_bp.route("/docs")
def swagger_ui():
    """Serves the interactive Swagger UI documentation page."""
    return Response(SWAGGER_HTML, mimetype="text/html")