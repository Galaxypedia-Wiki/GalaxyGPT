async function submitthedataidk() {
    // var page = document.getElementById("pageinputthing").value;
    var promptinput = document.getElementById("ballin")
    var prompt = promptinput.value;
    var submitbutton = document.getElementById("tomfoolery");
    var responsesection = document.getElementById("response");

    promptinput.disabled = true;
    submitbutton.disabled = true;
    submitbutton.style.cursor = "not-allowed";
    responsesection.innerHTML = "Thinking...";
    

    var data = await fetch("/api/v1/ask", {
    method: "POST",
    body: JSON.stringify({prompt: prompt}),
    headers: {
        "Content-Type": "application/json"
    }
    });
    
    if (data.status != 200) {
        responsesection.innerHTML = "Error: " + (await data.json());
        promptinput.disabled = false;
        submitbutton.disabled = false;
        submitbutton.style.cursor = "pointer";
        promptinput.focus()
        return;
    }

    var datajson = await data.json();
    responsesection.innerHTML = datajson.answer;

    promptinput.disabled = false;
    submitbutton.disabled = false;
    submitbutton.style.cursor = "pointer";
    promptinput.focus()
}

async function animateTitle() {
    const title = document.getElementById("title");
    const titlecard = document.getElementById("titlecard");
    const subtitle = document.getElementById("subtitle");
    const sleep = (milliseconds) => { return new Promise(resolve => setTimeout(resolve, milliseconds)) }
    await sleep(100)

    title.style.opacity = 1;
    title.style.transform = "translateY(0px) scale(0.8)";

    await sleep(500);

    title.style.transform = "translateY(0px) scale(1)"
    title.style.letterSpacing = "5px";
    titlecard.style.boxShadow = "0px 8px 10px #404244";
    titlecard.style.border = "#404244 solid 1px"
    // title.style.textShadow = "0px 0px 10px #fff";
    subtitle.style.opacity = 1;


}

document.onload = animateTitle();

const amongus = document.getElementById("ballin");

amongus.addEventListener("keypress", (key) => {

    if (key.key === "Enter") {
        submitthedataidk();
    }
    
})