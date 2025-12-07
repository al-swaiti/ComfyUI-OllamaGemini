import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'

/**
 * Veo Video Extend Widget Extension
 * Adds video preview and upload functionality to VeoVideoExtend node
 */

function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("Tried to add callback to non-existent object")
        return;
    }
    if (property in object && object[property]) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            return callback.apply(this, arguments) ?? r
        };
    } else {
        object[property] = callback;
    }
}

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
    node?.graph?.setDirtyCanvas(true);
}

async function uploadFile(file, progressCallback) {
    try {
        const body = new FormData();
        const i = file.webkitRelativePath?.lastIndexOf('/') ?? -1;
        const subfolder = i > 0 ? file.webkitRelativePath.slice(0, i + 1) : '';
        const new_file = new File([file], file.name, {
            type: file.type,
            lastModified: file.lastModified,
        });
        body.append("image", new_file);
        if (subfolder) {
            body.append("subfolder", subfolder);
        }
        const url = api.apiURL("/upload/image")
        const resp = await new Promise((resolve) => {
            let req = new XMLHttpRequest()
            req.upload.onprogress = (e) => progressCallback?.(e.loaded / e.total)
            req.onload = () => resolve(req)
            req.open('post', url, true)
            req.send(body)
        })

        if (resp.status !== 200) {
            alert(resp.status + " - " + resp.statusText);
        }
        return resp
    } catch (error) {
        alert(error);
    }
}

function addVideoPreview(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        var element = document.createElement("div");
        const previewNode = this;
        
        var previewWidget = this.addDOMWidget("videopreview", "preview", element, {
            serialize: false,
            hideOnZoom: false,
            getValue() {
                return element.value;
            },
            setValue(v) {
                element.value = v;
            },
        });
        
        previewWidget.computeSize = function(width) {
            if (this.aspectRatio && !this.parentEl.hidden) {
                let height = (previewNode.size[0] - 20) / this.aspectRatio + 10;
                if (!(height > 0)) {
                    height = 0;
                }
                this.computedHeight = height + 10;
                return [width, height];
            }
            return [width, -4];
        }
        
        // Prevent context menu issues
        element.addEventListener('contextmenu', (e) => {
            e.preventDefault()
            return app.canvas._mousedown_callback(e)
        }, true);
        
        previewWidget.value = { hidden: false, paused: false, params: {} }
        previewWidget.parentEl = document.createElement("div");
        previewWidget.parentEl.className = "veo_preview";
        previewWidget.parentEl.style['width'] = "100%"
        element.appendChild(previewWidget.parentEl);
        
        previewWidget.videoEl = document.createElement("video");
        previewWidget.videoEl.controls = true;
        previewWidget.videoEl.loop = true;
        previewWidget.videoEl.muted = true;
        previewWidget.videoEl.style['width'] = "100%"
        previewWidget.videoEl.style['borderRadius'] = "4px"
        
        previewWidget.videoEl.addEventListener("loadedmetadata", () => {
            previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
            fitHeight(previewNode);
        });
        
        previewWidget.videoEl.addEventListener("error", () => {
            previewWidget.parentEl.hidden = true;
            fitHeight(previewNode);
        });
        
        previewWidget.videoEl.onmouseenter = () => {
            previewWidget.videoEl.muted = false;
        };
        
        previewWidget.videoEl.onmouseleave = () => {
            previewWidget.videoEl.muted = true;
        };
        
        previewWidget.parentEl.appendChild(previewWidget.videoEl)
        
        this.updateVideoPreview = (filename, type = "input") => {
            if (!filename) {
                previewWidget.parentEl.hidden = true;
                fitHeight(this);
                return;
            }
            
            previewWidget.parentEl.hidden = false;
            let extension_index = filename.lastIndexOf(".");
            let extension = filename.slice(extension_index + 1);
            let format = "video/" + extension;
            
            let params = {
                filename: filename,
                type: type,
                format: format,
                timestamp: Date.now()
            };
            
            previewWidget.videoEl.src = api.apiURL('/view?' + new URLSearchParams(params));
            previewWidget.videoEl.autoplay = true;
        };
    });
}

function addUploadWidget(nodeType, widgetName) {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const node = this;
        const pathWidget = this.widgets.find((w) => w.name === widgetName);
        const fileInput = document.createElement("input");
        
        chainCallback(this, "onRemoved", () => {
            fileInput?.remove();
        });
        
        const accept = ["video/webm", "video/mp4", "video/x-matroska", "video/quicktime", "video/x-msvideo"];
        
        async function doUpload(file) {
            let resp = await uploadFile(file, (p) => node.progress = p)
            node.progress = undefined
            if (resp.status != 200) {
                return false
            }
            const filename = JSON.parse(resp.responseText).name;
            pathWidget.options.values.push(filename);
            pathWidget.value = filename;
            if (pathWidget.callback) {
                pathWidget.callback(filename)
            }
            return true
        }
        
        Object.assign(fileInput, {
            type: "file",
            accept: accept.join(','),
            style: "display: none",
            onchange: async () => {
                if (fileInput.files.length) {
                    return await doUpload(fileInput.files[0])
                }
            },
        });
        
        // Allow drag and drop
        this.onDragOver = (e) => !!e?.dataTransfer?.types?.includes?.('Files')
        this.onDragDrop = async function(e) {
            if (!e?.dataTransfer?.types?.includes?.('Files')) {
                return false
            }
            const item = e.dataTransfer?.files?.[0]
            if (item?.type?.startsWith('video/')) {
                return await doUpload(item)
            }
            return false
        }
        
        document.body.append(fileInput);
        
        let uploadWidget = this.addWidget("button", "📁 Upload Video", "upload", () => {
            app.canvas.node_widget = null;
            fileInput.click();
        });
        uploadWidget.options.serialize = false;
        
        // Wire up the path widget callback to update preview
        chainCallback(pathWidget, "callback", (value) => {
            if (node.updateVideoPreview) {
                node.updateVideoPreview(value, "input");
            }
        });
        
        // Initial preview if value exists
        if (pathWidget.value) {
            setTimeout(() => {
                if (node.updateVideoPreview) {
                    node.updateVideoPreview(pathWidget.value, "input");
                }
            }, 100);
        }
    });
}

app.registerExtension({
    name: "OllamaGemini.VeoWidgets",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // VeoVideoExtend no longer has video_file dropdown - it requires video_uri input
        // So we don't add upload widget for it anymore
        
        // Add video preview and upload for VeoLoadVideo
        if (nodeData?.name === "VeoLoadVideo") {
            addVideoPreview(nodeType);
            addUploadWidget(nodeType, "video_file");
        }
    },
    
    async setup() {
        console.log("🎬 VEO Widgets extension loaded");
    }
});
