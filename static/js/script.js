function sure(id){
    if(!confirm("Are you sure you want to return?\n"+
        "If you do, you will lose all your progress!")){
            preventDefault;
        }
        else{
            window.location.href = "/"+ id;
        }
}
