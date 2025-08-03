#!/bin/bash
set -e

# Configuration
ENDPOINT_ID=""
API_KEY=""
TEST_FILE="test_input.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    echo "Usage: $0 [OPTIONS] ENDPOINT_ID API_KEY"
    echo ""
    echo "Test the deployed Wan2.2 RunPod serverless endpoint"
    echo ""
    echo "Options:"
    echo "  -f, --file FILE      Test input file (default: test_input.json)"
    echo "  -s, --sync           Use synchronous endpoint (default)"
    echo "  -a, --async          Use asynchronous endpoint"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Arguments:"
    echo "  ENDPOINT_ID          RunPod endpoint ID"
    echo "  API_KEY              RunPod API key"
    echo ""
    echo "Examples:"
    echo "  $0 abcd1234 your-api-key                    # Test with default input"
    echo "  $0 -f test_input_complex.json abcd1234 key  # Test with complex input"
    echo "  $0 -a abcd1234 your-api-key                 # Test asynchronously"
}

# Default values
SYNC_MODE=true
ENDPOINT_TYPE="runsync"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            TEST_FILE="$2"
            shift 2
            ;;
        -s|--sync)
            SYNC_MODE=true
            ENDPOINT_TYPE="runsync"
            shift
            ;;
        -a|--async)
            SYNC_MODE=false
            ENDPOINT_TYPE="run"
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            print_error "Unknown option $1"
            show_usage
            exit 1
            ;;
        *)
            if [[ -z "$ENDPOINT_ID" ]]; then
                ENDPOINT_ID="$1"
            elif [[ -z "$API_KEY" ]]; then
                API_KEY="$1"
            else
                print_error "Too many arguments"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$ENDPOINT_ID" ]]; then
    print_error "Endpoint ID is required"
    show_usage
    exit 1
fi

if [[ -z "$API_KEY" ]]; then
    print_error "API key is required"
    show_usage
    exit 1
fi

# Check if test file exists
if [[ ! -f "$TEST_FILE" ]]; then
    print_error "Test file not found: $TEST_FILE"
    exit 1
fi

# Construct API URL
API_URL="https://api.runpod.ai/v2/${ENDPOINT_ID}/${ENDPOINT_TYPE}"

print_status "Testing RunPod endpoint..."
print_status "Endpoint ID: $ENDPOINT_ID"
print_status "Mode: $([[ $SYNC_MODE == true ]] && echo "Synchronous" || echo "Asynchronous")"
print_status "Test file: $TEST_FILE"
print_status "API URL: $API_URL"

# Check if curl is available
if ! command -v curl &> /dev/null; then
    print_error "curl is required but not installed"
    exit 1
fi

# Check if jq is available for JSON parsing
if ! command -v jq &> /dev/null; then
    print_warning "jq is not installed - output will not be formatted"
    USE_JQ=false
else
    USE_JQ=true
fi

print_status "Sending request..."

# Record start time
START_TIME=$(date +%s)

# Make the API request
if [[ $USE_JQ == true ]]; then
    RESPONSE=$(curl -s -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d @"$TEST_FILE")
    
    # Check if response is valid JSON
    if echo "$RESPONSE" | jq . >/dev/null 2>&1; then
        echo "$RESPONSE" | jq .
        
        # Check for success
        STATUS=$(echo "$RESPONSE" | jq -r '.status // empty')
        if [[ "$STATUS" == "success" ]]; then
            print_success "Request completed successfully!"
            
            # Extract video path if available
            VIDEO_PATH=$(echo "$RESPONSE" | jq -r '.video_path // empty')
            if [[ -n "$VIDEO_PATH" ]]; then
                print_status "Video saved to: $VIDEO_PATH"
            fi
            
            # Extract timing information if available
            GENERATION_TIME=$(echo "$RESPONSE" | jq -r '.metadata.timing.total_request_time_formatted // empty')
            if [[ -n "$GENERATION_TIME" ]]; then
                print_status "Generation time: $GENERATION_TIME"
            fi
            
        elif [[ "$STATUS" == "error" ]]; then
            ERROR_MSG=$(echo "$RESPONSE" | jq -r '.message // "Unknown error"')
            print_error "Request failed: $ERROR_MSG"
            exit 1
        else
            print_warning "Request status unclear"
        fi
    else
        print_error "Invalid JSON response:"
        echo "$RESPONSE"
        exit 1
    fi
else
    # Without jq, just show raw response
    curl -X POST "$API_URL" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        -d @"$TEST_FILE"
    echo ""
fi

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

print_success "Test completed in ${TOTAL_TIME} seconds"

# Show additional info for async mode
if [[ $SYNC_MODE == false ]]; then
    echo ""
    print_status "For async requests, you can check the status with:"
    echo "curl -H \"Authorization: Bearer $API_KEY\" https://api.runpod.ai/v2/\$JOB_ID/status"
fi 