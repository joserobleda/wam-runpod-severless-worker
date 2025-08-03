#!/bin/bash
set -e

# Configuration
IMAGE_NAME="wan22-t2v-diffusers"
REGISTRY="docker.io"
DOCKERFILE="Dockerfile"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS] USERNAME"
    echo ""
    echo "Deploy Wan2.2 T2V model to RunPod serverless"
    echo ""
    echo "Options:"
    echo "  -t, --tag TAG        Image tag (default: latest)"
    echo "  -p, --push           Push to registry after building"
    echo "  -f, --force          Force rebuild without cache"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Arguments:"
    echo "  USERNAME             Docker registry username"
    echo ""
    echo "Examples:"
    echo "  $0 myusername                    # Build image"
    echo "  $0 -p myusername                 # Build and push image"
    echo "  $0 -t v1.0.0 -p myusername       # Build and push with custom tag"
    echo "  $0 -f -p myusername              # Force rebuild and push"
}

# Default values
TAG="latest"
PUSH=false
FORCE=false
USERNAME=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -f|--force)
            FORCE=true
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
            if [[ -z "$USERNAME" ]]; then
                USERNAME="$1"
            else
                print_error "Too many arguments"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate username
if [[ -z "$USERNAME" ]]; then
    print_error "Username is required"
    show_usage
    exit 1
fi

# Construct full image name
FULL_IMAGE_NAME="${REGISTRY}/${USERNAME}/${IMAGE_NAME}:${TAG}"

print_status "Starting deployment process..."
print_status "Image: ${FULL_IMAGE_NAME}"
print_status "Push to registry: ${PUSH}"
print_status "Force rebuild: ${FORCE}"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    print_error "Docker is not running or not accessible"
    exit 1
fi

# Check if Dockerfile exists
if [[ ! -f "$DOCKERFILE" ]]; then
    print_error "Dockerfile not found: $DOCKERFILE"
    exit 1
fi

# Build arguments
BUILD_ARGS=(
    "build"
    "--platform" "linux/amd64"
    "-t" "$FULL_IMAGE_NAME"
)

# Add no-cache flag if force rebuild
if [[ "$FORCE" == true ]]; then
    BUILD_ARGS+=("--no-cache")
    print_warning "Building without cache (force rebuild)"
fi

# Add current directory as build context
BUILD_ARGS+=(".")

print_status "Building Docker image..."
print_status "Command: docker ${BUILD_ARGS[*]}"

# Build the image
if docker "${BUILD_ARGS[@]}"; then
    print_success "Docker image built successfully"
else
    print_error "Failed to build Docker image"
    exit 1
fi

# Get image size
IMAGE_SIZE=$(docker images --format "table {{.Size}}" "$FULL_IMAGE_NAME" | tail -1)
print_status "Image size: $IMAGE_SIZE"

# Push to registry if requested
if [[ "$PUSH" == true ]]; then
    print_status "Pushing image to registry..."
    
    # Check if logged in to Docker registry
    if ! docker info | grep -q "Username:"; then
        print_warning "Not logged in to Docker registry"
        print_status "Attempting to log in..."
        
        if ! docker login; then
            print_error "Failed to log in to Docker registry"
            exit 1
        fi
    fi
    
    # Push the image
    if docker push "$FULL_IMAGE_NAME"; then
        print_success "Image pushed successfully to $FULL_IMAGE_NAME"
    else
        print_error "Failed to push image"
        exit 1
    fi
fi

# Show final instructions
echo ""
print_success "Deployment process completed!"
echo ""
print_status "Next steps:"
echo "1. Copy this image URL to use in RunPod:"
echo "   ${FULL_IMAGE_NAME}"
echo ""
echo "2. Create a RunPod serverless endpoint with:"
echo "   - Docker Image: ${FULL_IMAGE_NAME}"
echo "   - GPU: NVIDIA A100 (40GB or 80GB)"
echo "   - Network Volume: Attached for model storage"
echo "   - Environment Variables (optional):"
echo "     HF_HUB_CACHE=/runpod-volume/.cache/huggingface"
echo ""
echo "3. Test the endpoint with the provided test files:"
echo "   - test_input.json (simple test)"
echo "   - test_input_complex.json (complex test)"
echo ""
print_status "Happy video generating! 🎬🎉" 