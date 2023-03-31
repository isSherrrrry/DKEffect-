var graph;

var defaultX = 0, defaultY = 1;
var x0 = null, x0old = null, x1 = null, dims = [];
var attrNo = null, attr = null, attr2 = [], index = 0;
var X = [], Y = [];
var loaddata = [];
var istxtdata = false;
var colorScale = d3.scale.category10();
var label = { DEFAULT: "Un-Assign", SUV: "SUV", S: "Sedan", SC: "Sports Car", W: "Wagon", M: "Minivan" };
var positions = [label.DEFAULT, label.SUV, label.S, label.SC, label.W, label.M];
var colorMap = { "Un-Assign": "#7f7f7f", "SUV": colorScale(0), "Sedan": colorScale(1), "Sports Car": colorScale(2), "Wagon": colorScale(3), "Minivan": colorScale(4) };
var defaultColor = colorScale(5);
defaultColor = colorScale(6);
defaultColor = colorScale(7);
var positionDescriptions;// = positionDescriptions1;
var activePosition = "none";
var playerPositionMap = {};
var helpMouseoverStart = 0;
var helpMouseoverEnd = 0;
var allData2 = [];


function getInitX() {
	// var sel=document.getElementById('initX');
	// var sid=sel.selectedIndex;
	// var defaultX = sel[sid].innerHTML
	var mySelectX = document.getElementById('initX');
	var defaultValueX = mySelectX.options[mySelectX.selectedIndex].text;

	console.log(defaultValueX);
	return defaultValueX

	// alert(oOptVal);
}
function getInitY() {
	var mySelectY = document.getElementById('initY');
	var defaultValueY = mySelectY.options[mySelectY.selectedIndex].text;
	console.log(defaultValueY);
	return defaultValueY
}


//Gets called when the page is loaded.
function init() {
	// defaultY = defaultY;
	// defaultX = mySelectX.options[indexX].text;
	// defaultY = mySelectY.options[indexY].text;
	// defaultX =$("#initX").find("option:selected").text();
	// defaultY =$("#initY").find("option:selected").text();
	defaultX = $("#initX :selected").text();
	defaultY = $("#initY :selected").text();

	// defaultX = localStorage.getItem("defaultX");
	// defaultY = localStorage.getItem("defaultY");
	$("#area1").show();
	$("#area2").show();
	$("#area3").show();
	$("#dialog").hide();
	loadData();
}

function loadData() {
	// Get input data
	d3.csv('data/practice_removed.csv', function (data) { // total 15 points for practice
		var loaddata = jQuery.extend(true, [], data);
		for (var i = 0; i < loaddata.length; i++) {
			delete loaddata[i]["Car"];
			delete loaddata[i]["Sedan"];
			delete loaddata[i]["Sports Car"];
			delete loaddata[i]["SUV"];
			delete loaddata[i]["Wagon"];
			delete loaddata[i]["Minivan"];

			playerPositionMap[loaddata[i]["Car Anonymized"]] = "none";
			loaddata[i]["coord"] = {};
		}
		attr = Object.keys(loaddata[0]);
		attr.pop(); // remove "coord" from attribute list
		attr.pop(); // remove "Name" from attribute list
		attrNo = attr.length;

		// set default user labels
		loaddata.forEach(function (d) {
			d["coord"]["userlabel"] = label.DEFAULT;
		});

		for (var i = 0; i < attrNo; i++) {
			if (attr[i] != "Name" && attr[i] != "coord") {
				var tmpmax = d3.max(loaddata, function (d) { return +d[attr[i]]; });
				var tmpmin = d3.min(loaddata, function (d) { return +d[attr[i]]; });

				// jitter the value by adding a random number up to +/- 3% of scale
				function jitter(val, attribute) {
					var mult = 0.03;
					var noise = Math.random() * mult * (tmpmax - tmpmin);

					// determine whether to add or subtract
					var sign = Math.round(Math.random());
					if (sign == 0) return val - noise;
					else return val + noise;
				}

				loaddata.forEach(function (d) {
					d["coord"][attr[i]] = jitter(+d[attr[i]], attr[i]);// jitter((+d[attr[i]]-tmpmin)/(tmpmax-tmpmin), attr[i]);
				});
			}
		}

		// determine which condition the user follows and which set of descriptions should be used
		// if (window.localStorage.getItem("whichCondition") == 1) positionDescriptions = positionDescriptions1;
		// else positionDescriptions = positionDescriptions2;
		//positionDescriptions = positionDescriptionsDemo;

		// initialize rapid7
	
		ial.init(loaddata, 0, ["coord", "Name"], "exclude", -1, 1);
		console.log("ial initialized");

		console.log(loaddata);
		// load the vis
		loadVis(loaddata);
		allData2.push(loaddata);
	});
}

//Main function
function loadVis(data) {
	drawScatterPlot(data);
	//updateBias(true);

	for (var i = 0; i < attrNo; i++) {
		dims[i] = attr[i];
	}
	drawParaCoords(data, dims);
	tabulate(data[0], 'empty');
	addHelp();
	addClassificationControls(data);
	addCustomAxisDropDownControls();

	// LE.log("Hello, logger!");
	// console.log("Hello, logger!");
}

function addCustomAxisDropDownControls() {
	$("#cbX").on('change', function () {
		if ($(this).val() != "Custom Axis") {
			$("#cbX option[value='Custom Axis']").remove();
		}
	});
}

function addHelp() {
	var tooltipText = '<div class="qtip-dark"><b>Task:</b> Your task is to classify all of the points in the scatterplot. Each <i>circle</i> in the scatterplot represents a <i>vehicle</i>. Color each circle according to the <i>type</i> of vehicle you think it is.';
	tooltipText += '<br><br>';
	tooltipText += '<b>Interactions:</b> <ul>';
	tooltipText += '<li><i>See Details</i> about a point by <i>Hovering</i> over it. Details will be shown in the text on the right.</li>';
	tooltipText += '<li><i>Activate a Type</i> by <i>Clicking</i> on a colored circle on the right.</li>';
	tooltipText += '<li><i>Deactivate a Type</i> by <i>Double Clicking</i> on any colored circle on the right.</li>';
	tooltipText += '<li><i>Classify a Point</i> on the scatterplot by <i>Clicking</i> on it while its vehicle type is activated.</li>';
	tooltipText += '<li><i>Un-Assign a Point</i> on the scatterplot by <i>Clicking</i> on it while "Un-Assign" is activated.</li>';
	tooltipText += '<li><i>Change the Axes</i> by <i>Selecting</i> a new variable from the drop-down on the X or Y axes.</li>';
	tooltipText += '</ul>';
	tooltipText += '<br>';
	tooltipText += 'Try to classify all points in the scatterplot to complete the study. When ready to continue, check the box in bottom right and press <strong><i>Continue</i></strong> to proceed to the next phase of the study.';
	tooltipText += '</div>';
	$("#helpButton").qtip({
		content: {
			title: 'Help:',
			text: tooltipText
		},
		style: {
			width: 500,
			classes: 'qtip-dark'
		}
	});
	d3.select("#helpButton").on("mouseover", function () {
		helpMouseoverStart = new Date();
	}).on("mouseout", function () {
		helpMouseoverEnd = new Date();
		// units are seconds -- mouseover time will always encapsulate drag time as well
		var elapsedTime = (helpMouseoverEnd - helpMouseoverStart) / 1000;

		// get the x,y coordinates of all the points on the graph to log
		var data_locations = [];
		d3.select("#SC").selectAll("circle").each(function (d) {
			var pt_log = { car: d.Name, x: d.x, y: d.y, cx: +this.getAttribute("cx"), cy: +this.getAttribute("cy") };
			data_locations.push(pt_log);
		});

		ial.logging.log('help', undefined, 'HelpHover', { 'level': 'INFO', 'eventType': 'help_hover', 'elapsedTime': elapsedTime, 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'data_locations': data_locations });
		// LE.log(JSON.stringify(ial.logging.peek()));
		R7Insight.log(JSON.stringify(ial.logging.peek()));
		// testing
		console.log(JSON.stringify(ial.logging.peek()));
	});
}

function addClassificationControls(data) {
	var svgClassContainer = d3.select("#datapanel2")
		.append("svg")
		.attr("width", 150)
		.attr("height", 160);
	var categoryGroups = svgClassContainer.selectAll("circle")
		.data(positions)
		.enter().append("g")
		.attr("transform", function (d, i) { return "translate(30, " + (24 * (i + 1)) + ")"; })
		.on("click", function (d) {
			//
			activePosition = d;
			d3.selectAll(".category").classed("categoryClicked", false);
			d3.select(this).select("circle").classed("categoryClicked", true);
			/* ---------- disabling image display ---------- */
			// $("#datapanel3").html("<h5 style=\"color:" + colorMap[d] + "; stroke: 1px black;\">" + d + ":</h5> <img src=" + positionDescriptions[d] + " width=304 height=228>");

			// get the x,y coordinates of all the points on the graph to log
			var data_locations = [];
			d3.select("#SC").selectAll("circle").each(function (d) {
				var pt_log = { car: d.Name, x: d.x, y: d.y, cx: +this.getAttribute("cx"), cy: +this.getAttribute("cy") };
				data_locations.push(pt_log);
			});

			ial.logging.log('category_' + activePosition, undefined, 'CategoryClick', { 'level': 'INFO', 'eventType': 'category_click', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'data_locations': data_locations });

			//LE.log(JSON.stringify(ial.logging.peek()));
			R7Insight.log(JSON.stringify(ial.logging.peek()));
		}).on("dblclick", function () {
			activePosition = "none";
			d3.selectAll(".category").classed("categoryClicked", false);
			$("#datapanel3").html("");

			// get the x,y coordinates of all the points on the graph to log
			var data_locations = [];
			d3.select("#SC").selectAll("circle").each(function (d) {
				var pt_log = { car: d.Name, x: d.x, y: d.y, cx: +this.getAttribute("cx"), cy: +this.getAttribute("cy") };
				data_locations.push(pt_log);
			});

			ial.logging.log('category_' + activePosition, undefined, 'CategoryDoubleClick', { 'level': 'INFO', 'eventType': 'category_double_click', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'data_locations': data_locations });
			// LE.log(JSON.stringify(ial.logging.peek()));
			R7Insight.log(JSON.stringify(ial.logging.peek()));
		});
	var categoryCircles = categoryGroups.append("circle")
		.attr("cx", 0)
		.attr("cy", function (d, i) { return 0; })
		.attr("r", 10)
		.classed("category", true)
		.style("fill", function (d) { return colorMap[d]; });
	var categoryText = categoryGroups.append("text")
		.attr("dx", function (d) { return 20; })
		.text(function (d) { return d; })
		.attr("dominant-baseline", "middle");


	// listen for user to hit Continue
	// var x = Math.floor((Math.random() * 10) + 1);
	// console.log(x);
	$("#continueButton").click(function () {
		// var x = Math.floor((Math.random() * 10) + 1);
		// console.log(x);
		if (document.getElementById('doneCheck').checked) {
			var allClassified = true;
			var howMany = 0;
			for (var i = 0; i < data.length; i++) {
				if (data[i]["coord"]["userlabel"] == label.DEFAULT) {
					allClassified = false;
					howMany++;
				}
			}
			// var prevURL = document.referrer;
			if (!allClassified) {
				if ((data.length - howMany) <= (data.length * 1)) {
					var userResp = alert("Please classify more points.");
				} else {
					var userResp = confirm(howMany + " out of " + data.length + " points have not been classified yet. Are you sure you want to continue?");
					if (userResp) { // user wants to continue anyway
						// var x = Math.floor((Math.random() * 10) + 1);
						// console.log(x);
						// document.getElementById('continueButton').onclick = function () {
							// if (x % 2 == 0) {
								localStorage.setItem('first_task', 'credit');
								var w = window.open('postsurvey.html', '_self');
								var filename = window.localStorage.getItem("userId") + "car";
								console.save(allData2, filename);
							// } else {
							// 	localStorage.setItem('first_task', 'dog');
							// 	var w = window.open('custom_axis_dog.html', '_self');
							// };
						// };
						//disable back button
						history.go(1);
					}
				}
			} else {
				// var x = Math.floor((Math.random() * 10) + 1);
				// console.log(x);
				// document.getElementById('continueButton').onclick = function () {
					// if (x % 2 == 0) {
						localStorage.setItem('first_task', 'credit');
						var w = window.open('postsurvey.html', '_self');
						var filename = window.localStorage.getItem("userId") + "car";
						console.save(allData2, filename);
					// } else {
					// 	localStorage.setItem('first_task', 'dog');
					// 	var w = window.open('custom_axis_credit.html', '_self');
					// };
				// };
				//disable back button
				history.go(1);
				window.localStorage.setItem("userId", window.localStorage.getItem("userId"));
				window.localStorage.setItem("whichCondition", window.localStorage.getItem("whichCondition"));
			}

		} else alert("Check the box to certify you are finished with the task.");
	});
}

function drawScatterPlot(data) {

	// heterogeneous data
	initdim1 = attr.indexOf(localStorage.getItem("defaultX_car")), initdim2 = attr.indexOf(localStorage.getItem("defaultY_car")); // 1, 2 = height, weight
	// initdim1 = attr.indexOf('HP'), initdim2 = attr.indexOf('Weight');
	data.forEach(function (d) { d.x = d["coord"][attr[initdim1]]; d.y = d["coord"][attr[initdim2]]; });
	graph = new SimpleGraph("scplot", data, {
		"xlabel": attr[initdim1],
		"ylabel": attr[initdim2],
		"init": true
	});

	var V1 = {}, V2 = {};
	for (var i = 0; i < attrNo; i++) {
		X[i] = { "attr": attr[i], "value": 0, "changed": 0, "error": 0 };
		Y[i] = { "attr": attr[i], "value": 0, "changed": 0, "error": 0 };
		V1[attr[i]] = 0;
		V2[attr[i]] = 0;
	}

	V1[attr[initdim1]] = 1;
	V2[attr[initdim2]] = 1;

	// get the x,y coordinates of all the points on the graph to log
	var data_locations = [];
	d3.select("#SC").selectAll("circle").each(function (d) {
		var pt_log = { car: d.Name, x: d.x, y: d.y, cx: +this.getAttribute("cx"), cy: +this.getAttribute("cy") };
		data_locations.push(pt_log);
	});

	//update IAL weight vector
	ial.usermodel.setAttributeWeightVector(V1, true, { 'level': 'INFO', 'eventType': 'set_attribute_weight_vector_init', 'whichAxis': 'X', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'data_locations': data_locations });
	R7Insight.log(JSON.stringify(ial.logging.peek()));
	//LE.log(JSON.stringify(ial.logging.peek()));
	ial.usermodel.setAttributeWeightVector(V2, true, { 'level': 'INFO', 'eventType': 'set_attribute_weight_vector_init', 'whichAxis': 'Y', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'data_locations': data_locations });
	// LE.log(JSON.stringify(ial.logging.peek()));
	R7Insight.log(JSON.stringify(ial.logging.peek()));

	X[initdim1]["value"] = 1;
	Y[initdim2]["value"] = 1;
	document.getElementById("cbX").selectedIndex = initdim1;
	document.getElementById("cbY").selectedIndex = initdim2;

	xaxis = new axis("#scplot", X, "X", {
		"width": graph.size.width - dropSize * 2,
		"height": graph.padding.bottom - 40,
		"padding": { top: graph.padding.top + graph.size.height + 40, right: 0, left: graph.padding.left + dropSize + 10, bottom: 0 }
	});
}