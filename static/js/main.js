    // Load the Visualization API and the piechart package.
      google.charts.load('current', {'packages':['corechart']});

      // Set a callback to run when the Google Visualization API is loaded.
      google.charts.setOnLoadCallback(drawChart);

      // Callback that creates and populates a data table,
      // instantiates the pie chart, passes in the data and
      // draws it.
      function drawChart() {

      // Create the data table.
      var data = google.visualization.arrayToDataTable([
          ['Year', 'Predicted', 'Actual'],
          ['2004',  1000,      900],
          ['2005',  1170,      1250],
          ['2006',  660,       700],
          ['2007',  1030,      1100]
      ]);
      // var height = $('body').height();
      // var width = $('body').width();
      //
      // if (height > width){
      //     var chart_height = height*0.7;
      //     var chart_width = width;
      //     $('#result').css('width', '100%');
      //     $('#chart_div').css('width', '100%');
      //
      // }
      // else {
      //     var chart_width = width*0.46;
      //     var chart_height = chart_width*0.66;
      //     $('#result').css('width', chart_width);
      //     $('#result').css('height', chart_height);
      // }
      // console.log(height, width, chart_height, chart_width);
      // Set chart options
      var options = {'title':'Load History of Gandhinagar',
                    // 'height': chart_height,
                    // 'width': chart_width,
                    'animation':{
                        'duration': 1000,
                        'easing': 'out',
                        'startup': true
                    }};

      var chart = new google.visualization.LineChart(document.getElementById('map'));
      chart.draw(data, options);
    }