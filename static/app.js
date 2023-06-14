var script = document.createElement('script');
script.src = 'https://code.jquery.com/jquery-3.6.3.min.js';

var enforeMutualExcludedCheckBox = function(group){
    return function() {
      var isChecked= $(this).prop("checked");
      $(group).prop("checked", false);
      $(this).prop("checked", isChecked);
    }
};

$(".selectanswer").click(enforeMutualExcludedCheckBox(".selectanswer"));
$(".selectanswer_feature").click(enforeMutualExcludedCheckBox(".selectanswer_feature"));


let eraseButton = document.querySelector('.erase')

class Data {
    constructor() {
       this.state = false;
       this.messages = [];
    } 

    setting()
    {
        eraseButton.addEventListener('click', () => this.eraseState())
    }


     eraseState()
    {    /* check the address of this server before do any fetch or try to access it from 0.0.0.0:8080 */
         fetch('http://10.50.120.157:8080/erase', {
            method: 'POST',
            mode: 'cors',
            headers: {
              'Content-Type': 'application/json'
            },
          })
          .then(r => r.json())
          .then(r => {
          let msg_ini = { name: "User", message: r.answer };
          this.messages.push(msg_ini);
          textField.value = ''
          }).catch((error) => {
            console.error('Error:', error);
            textField.value = ''
          }); 
      }

}
const DATA = new Data();
DATA.setting()
