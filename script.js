function replace_str_to_img(input, img_list) {
    let count = 0;
    return input.replace(/\(img\)\[(.*?)\]/g, function(match, name) {
        if (count < img_list.length) {
            let img_html = `<img src="${img_list[count]}" alt="${name}" loading="lazy">`;
            count++;
            return img_html;
        } else {
            return match;
        }
    });
}
function generate_gift_pack(data,D){
    let inner_html='';
    var name = '';
    var t = '';
    var url = '';
    var code = '';
    var mark = '';
    let img = [];
    
    if (D==='expired'){
      var coc_gift_pack = data.coc_gift_pack_expired;
    }else{
      var coc_gift_pack = data.coc_gift_pack;
    }
    //console.log(data.coc_gift_pack);
    var keys = Object.keys(coc_gift_pack);
    if (keys.length === 0){
        inner_html = '<div>&emsp;當前無可用禮包</div>'
        return inner_html
    }
    for (var index in keys) {
        key = keys[index];
        name = key;
        pack_data = coc_gift_pack[name];
        
        t = pack_data.time;
        url = pack_data.url;
        code = pack_data.code;
        mark = pack_data.markdown;
        img = pack_data.img;
        //inner_html+='<span>123</span><br>';
        inner_html += `<span>&emsp;(${t}新增)`; //name
        inner_html += replace_str_to_img(mark,img)
        inner_html += `</span><br>&emsp;&emsp;`;
        inner_html += `<a href="${url}">`;
        if (D==='expired'){
          inner_html += `<del><span>${code}</span></del>`;
        }else{
          inner_html += `<span>${code}</span>`;
        }
        inner_html += '</a><br><br>'; 
    }
    //document.getElementById("csvTable").innerHTML = inner_html;
    return inner_html
}
function generate_info(data){
    let inner_html='';
    var name = '';
    var url = '';
    //console.log(data.info);
    var info_data = data.info;
    var keys = Object.keys(info_data);
    if (keys.length === 0){
        inner_html = '<div>&emsp;當前無任何情報</div>'
        return inner_html
    }
    for (var index in keys) {
        name = keys[index];
        url = info_data[name];
        inner_html += `&emsp;<a href="${url}"><span>${name}</span></a><br><br>`; 
    }
    //document.getElementById("csvTable").innerHTML = inner_html;
    return inner_html
}
function generate_Useful_web(data){
    let inner_html='';
    var name = '';
    var name2 = '';
    var img = '';
    var url = '';
    //console.log(data.useful_web);
    var useful_web_data = data.useful_web;
    var keys = Object.keys(useful_web_data);
    if (keys.length === 0){
        inner_html = '<div>&emsp;None</div>'
        return inner_html
    }
    for (var index in keys) {
        name = keys[index];
        
        url = useful_web_data[name].url;
        img = useful_web_data[name].img;
        name2 = useful_web_data[name].name;
        inner_html += `<span>&emsp;${name}</span><br>&emsp;&emsp;`; 
        inner_html += `<a href="${url}">`
        if (img!==''){
          inner_html += `<img src="${img}" loading="lazy">`
        }
        inner_html += `<span>${name2}</span>`
        inner_html += '</a><br><br>'
    }
    //document.getElementById("csvTable").innerHTML = inner_html;
    return inner_html
}