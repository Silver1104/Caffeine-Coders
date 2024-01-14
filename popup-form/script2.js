$('button').hover(function(){
    var $this = $(this);
    var $prevAll = $(this).prevAll();
    
    var className = $this.attr("class") + "-hover";
    
    $this.addClass(className);
    $prevAll.addClass(className);
 }, function() {
    var $this = $(this);
    var $prevAll = $(this).prevAll();
    
    $this.removeClass("detractor-hover passive-hover promoter-hover");
    $prevAll.removeClass("detractor-hover passive-hover promoter-hover");
 });
//  $('button').click(function() {
//    // Add a specific click class to the clicked button and its previous siblings
//    var $this = $(this);
//    var $prevAll = $(this).prevAll();
//    $this.addClass("button-click");
//    $prevAll.addClass("button-click");
// });
$('button').click(function(){
   var $this = $(this);
   var $prevAll = $(this).prevAll();
   
   var className = $this.attr("class") + "-click";
   
   $this.addClass(className);
   $prevAll.addClass(className);
// }, function() {
//    var $this = $(this);
//    var $prevAll = $(this).prevAll();
   
//    $this.removeClass("detractor-hover passive-hover promoter-hover");
//    $prevAll.removeClass("detractor-hover passive-hover promoter-hover");
// }
});