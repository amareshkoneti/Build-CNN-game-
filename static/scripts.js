document.addEventListener("DOMContentLoaded", function () {
    const addLabelBtn = document.getElementById("add-label-btn");
    const labelInput = document.getElementById("label-input");
    const labelsContainer = document.getElementById("labels-container");
    const uploadSectionsContainer = document.getElementById("upload-sections-container");
    const uploadedImagesContainer = document.getElementById("uploaded-images-container");
  
    addLabelBtn.addEventListener("click", function () {
        const labelName = labelInput.value.trim();
        if (labelName === "") return;
  
        // Check if label section already exists
        let labelDiv = document.getElementById(`label-${labelName}`);
        if (!labelDiv) {
            labelDiv = document.createElement("div");
            labelDiv.classList.add("label-section");
            labelDiv.id = `label-${labelName}`;
  
            // Label title
            const labelTitle = document.createElement("h3");
            labelTitle.textContent = labelName;
  
            // Instruction message
            const labelMessage = document.createElement("p");
            labelMessage.textContent = `Please upload ${labelName} images for efficient building.`;
            labelMessage.classList.add("label-message");
            labelMessage.style.color = "red";
            labelMessage.style.fontWeight = "bold";
  
            // File input
            const fileInput = document.createElement("input");
            fileInput.type = "file";
            fileInput.classList.add("image-upload-input");
            fileInput.accept = "image/*";
            fileInput.multiple = true;
  
            // Upload button
            const uploadBtn = document.createElement("button");
            uploadBtn.textContent = "Upload";
            uploadBtn.classList.add("upload-btn");
  
            // Uploaded images container
            const uploadedImagesDiv = document.createElement("div");
            uploadedImagesDiv.classList.add("uploaded-images");
  
            // Append elements
            labelDiv.appendChild(labelTitle);
            labelDiv.appendChild(labelMessage); // Instruction message
            labelDiv.appendChild(fileInput);
            labelDiv.appendChild(uploadBtn);
            labelDiv.appendChild(uploadedImagesDiv);
  
            uploadSectionsContainer.appendChild(labelDiv);
  
            // Handle file upload
            uploadBtn.addEventListener("click", function () {
                const files = fileInput.files;
                if (files.length === 0) return;
  
                const formData = new FormData();
                formData.append("label", labelName);
  
                for (let i = 0; i < files.length; i++) {
                    formData.append("image", files[i]);
                }
  
                fetch("/upload", {
                    method: "POST",
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    alert(data.message);
                    displayUploadedImages(labelName, files);
                })
                .catch(error => console.error("Error:", error));
            });
        }
        labelInput.value = "";
    });
  
    function displayUploadedImages(label, files) {
        let labelDiv = document.getElementById(`uploaded-${label}`);
        if (!labelDiv) {
            labelDiv = document.createElement("div");
            labelDiv.classList.add("uploaded-label-section");
            labelDiv.id = `uploaded-${label}`;
            labelDiv.innerHTML = `<h3>${label}</h3><div class="image-container"></div>`;
            uploadedImagesContainer.appendChild(labelDiv);
        }
  
        const imageContainer = labelDiv.querySelector(".image-container");
  
        for (let i = 0; i < files.length; i++) {
            const img = document.createElement("img");
            img.src = URL.createObjectURL(files[i]);
            img.classList.add("uploaded-image");
            imageContainer.appendChild(img);
        }
    }
  });
  