/*const tabs= document.querySelectorAll('[data-tab-target]')
const tabContents= document.querySelectorAll('[data-tab-content]')
//Rama 2 chaval?
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const target= document.querySelector(tab.dataset.tabTarget)
        tabContents.forEach(tabContent => {
            tabContent.classList.remove('active')
        })
        tabs.forEach(tab => {
            tab.classList.remove('active')
        })
        tab.classList.add('active')
        target.classList.add('active')
    })
});*/

function deactivate(id) {
    //var clas= document.querySelector('input[name="class"]:checked').value;
    var clasN= document.getElementById(id);
    //var $radio = $('input[name="class"]:checked');
    //var id = $radio.attr('id');
    //console.log(clasN)

    var checkbox1 = document.getElementById(id);
    console.log("Checbox "+checkbox1)
    checkbox1.disabled = true;

    //var feat = document.getElementById("troncocomun");
    //var primerelemento = document.getElementById("1").value;

    //if(document.getElementById("1").value < 6) {
      //  var checkbox1 = document.getElementById("checkbox2");
       //checkbox1.disabled = true;
    //}
}