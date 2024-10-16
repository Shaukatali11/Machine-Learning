// frontend/static/js/main.js
document.getElementById("predict-form").onsubmit = async function(event) {
  event.preventDefault();
  
  const formData = new FormData(this);
  
  const response = await fetch("/predict", {
      method: "POST",
      body: formData
  });
  
  const result = await response.json();
  document.getElementById("result").innerText = "Prediction: " + result;
};
