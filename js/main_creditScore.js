var graph;

var defaultX = 0, defaultY = 1;
var x0 = null, x0old = null, x1 = null, dims = [];
var attrNo = null, attr = null, attr2 = [], index = 0;
var X = [], Y = [];
var loaddata = [];
var istxtdata = false;
var colorScale = d3.scale.category10();
var label = { DEFAULT: "Un-Assign", CR2: "Good", CR1: "Fair", CR0: "Poor"};
var positions = [label.DEFAULT, label.CR2, label.CR1, label.CR0];
var colorMap = { "Un-Assign": "#7f7f7f", "Good": "#609F3A", "Fair": "#E48024", "Poor": "#B82C2C"};
var defaultColor = colorScale(5);
defaultColor = colorScale(6);
defaultColor = colorScale(7);
var activePosition = "none";
var playerPositionMap = {};
var helpMouseoverStart = 0;
var helpMouseoverEnd = 0;


function getInitX() {

	var mySelectX = document.getElementById('initX');
	var defaultValueX = mySelectX.options[mySelectX.selectedIndex].text;

	console.log(defaultValueX);
	return defaultValueX

}
function getInitY() {
	var mySelectY = document.getElementById('initY');
	var defaultValueY = mySelectY.options[mySelectY.selectedIndex].text;
	console.log(defaultValueY);
	return defaultValueY
}


function init() {
	defaultX = $("#initX :selected").text();
	defaultY = $("#initY :selected").text();
	$("#area1").show();
	$("#area2").show();
	$("#area3").show();
	$("#dialog").hide();
	loadData();
}



function loadData() {
	// Get input data
	d3.csv('data/Credit_Score_clean_20_category.csv', function (data) { // demo50 dataset
		var loaddata = jQuery.extend(true, [], data);
		for (var i = 0; i < loaddata.length; i++) {
			loaddata[i]["name"] = "Name: " + loaddata[i]["Customer_id"];
			delete loaddata[i]["Customer ID"];
			delete loaddata[i]["Name"];
			playerPositionMap[loaddata[i]["name"]] = "none";
			loaddata[i]["coord"] = {};
		}
		console.log(loadData[0]);

		attr = Object.keys(loaddata[0]);
		attr.pop(); // remove "coord" from attribute list
		attr.pop(); // remove "Name" from attribute list
		attrNo = attr.length;

		// set default user labels
		loaddata.forEach(function (d) {
			d["coord"]["userlabel"] = label.DEFAULT;
		});

		for (var i = 0; i < attrNo; i++) {
			if (attr[i] != "name" && attr[i] != "coord") {
				var tmpmax = d3.max(loaddata, function (d) { return +d[attr[i]]; });
				var tmpmin = d3.min(loaddata, function (d) { return +d[attr[i]]; });

				// jitter the value by adding a random number up to +/- 3% of scale
				function jitter(val, attribute) {
					var mult = 0.03;
					if (attribute == "Height (Inches)") mult = 0.02;
					if (attribute == "Weight (Pounds)") mult = 0.01;
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

		R7Insight.init({
			token: 'd51a6564-1a20-435f-afef-41806a4bc4cb',
			region: 'us3'
		});
		// log something
		R7Insight.log("This is a log event");



		// initialize IAL
		ial.init(loaddata, 0, ["coord", "name"], "exclude", -1, 1);
		console.log("ial initialized");

		console.log(loaddata);
		// load the vis
		loadVis(loaddata);

	});
}

//Main function
function loadVis(data) {
	drawScatterPlot(data);

	for (var i = 0; i < attrNo; i++) {
		dims[i] = attr[i];
	}
	drawParaCoords(data, dims);
	tabulate(data[0], 'empty');
	addHelp();
	addClassificationControls(data);

	console.log("Hello, logger!");
}

// function addCustomAxisDropDownControls() {
// 	$("#cbX").on('change', function () {
// 		if ($(this).val() != "Custom Axis") {
// 			$("#cbX option[value='Custom Axis']").remove();
// 		}
// 	});
// }

function addHelp() {
	var tooltipText = '<div class="qtip-dark"><b>Task:</b> Your task is to classify all of the points in the scatterplot. Each <i>circle</i> in the scatterplot represents a <i> candidate and their credit score</i>. Color each circle according to the <i>breed</i> you think the credit score belongs to.';
	tooltipText += '<br><br>';
	tooltipText += '<b>Note:</b> Each attribute is evaluated on a scale ranging from <b>good</b> to <b>poor</b>';
	tooltipText += '<br><br>';
	tooltipText += '<b>Interactions:</b> <ul>';
	tooltipText += '<li><i>See Details</i> about a point by <i>Hovering</i> over it. Details will be shown in the text on the right.</li>';
	tooltipText += '<li><i>Activate a breed</i> by <i>Clicking</i> on a colored circle on the right.</li>';
	tooltipText += '<li><i>Deactivate a breed</i> by <i>Double Clicking</i> on any colored circle on the right.</li>';
	tooltipText += '<li><i>Classify a Point</i> on the scatterplot by <i>Clicking</i> on it while its breed is activated.</li>';
	tooltipText += '<li><i>Un-Assign a Point</i> on the scatterplot by <i>Clicking</i> on it while "Un-Assign" is activated.</li>';
	tooltipText += '<li><i>Change the Axes</i> by <i>Selecting</i> a new variable from the drop-down on the X or Y axes.</li>';
	tooltipText += '</ul>';
	tooltipText += '<br>';
	tooltipText += 'Try to classify all points in the scatterplot to complete the study. When ready to continue, check the box in bottom right and press set_attribute_weight_vector_drag to proceed to the next phase of the study.';
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
			var pt_log = { credit: d.name, x: d.x, y: d.y, cx: +this.getAttribute("cx"), cy: +this.getAttribute("cy") };
			data_locations.push(pt_log);
		});

		ial.logging.log('help', undefined, 'HelpHover', { 'level': 'INFO', 'taskType': 'credit', 'eventType': 'help_hover', 'elapsedTime': elapsedTime, 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'data_locations': data_locations });
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
				var pt_log = { credit: d.name, x: d.x, y: d.y, cx: +this.getAttribute("cx"), cy: +this.getAttribute("cy") };
				data_locations.push(pt_log);
			});

			ial.logging.log('category_' + activePosition, undefined, 'CategoryClick', { 'level': 'INFO', 'taskType': 'credit', 'eventType': 'category_click', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'data_locations': data_locations });

			//LE.log(JSON.stringify(ial.logging.peek()));
			R7Insight.log(JSON.stringify(ial.logging.peek()));
		}).on("dblclick", function () {
			activePosition = "none";
			d3.selectAll(".category").classed("categoryClicked", false);
			$("#datapanel3").html("");

			// get the x,y coordinates of all the points on the graph to log
			var data_locations = [];
			d3.select("#SC").selectAll("circle").each(function (d) {
				var pt_log = { credit: d.Name, x: d.x, y: d.y, cx: +this.getAttribute("cx"), cy: +this.getAttribute("cy") };
				data_locations.push(pt_log);
			});

			ial.logging.log('category_' + activePosition, undefined, 'CategoryDoubleClick', { 'level': 'INFO', 'taskType': 'credit', 'eventType': 'category_double_click', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'data_locations': data_locations });
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
	$("#continueButton").click(function () {
		if (document.getElementById('doneCheck').checked) {
			var allClassified = true;
			var howMany = 0;
			for (var i = 0; i < data.length; i++) {
				if (data[i]["coord"]["userlabel"] == label.DEFAULT) {
					allClassified = false;
					howMany++;
				}
			}
			var prevURL = document.referrer;
			if (localStorage.getItem("first_task") != 'credit') {
				if (!allClassified) {
					if ((data.length - howMany) <= (data.length * 0.8)) {
						var userResp = alert("Please classify more points.");
					} else {
						var userResp = confirm(howMany + " out of " + data.length + " points have not been classified yet. Are you sure you want to continue?");
						if (userResp) { // user wants to continue anyway
							var w = window.open("ending.html", "_self");
						}
					}
				} else {
					var w = window.open("ending.html", "_self");
					window.localStorage.setItem("userId", window.localStorage.getItem("userId"));
					window.localStorage.setItem("whichCondition", window.localStorage.getItem("whichCondition"));
				}
			} else {
				if (!allClassified) {
					if ((data.length - howMany) <= (data.length * 0.8)) {
						var userResp = alert("Please classify more points.");
					} else {
						var userResp = confirm(howMany + " out of " + data.length + " points have not been classified yet. Are you sure you want to continue?");
						if (userResp) { // user wants to continue anyway
							var w = window.open("ending.html", "_self");
						}
					}
				} else {
					var w = window.open("ending.html", "_self");
					window.localStorage.setItem("userId", window.localStorage.getItem("userId"));
					window.localStorage.setItem("whichCondition", window.localStorage.getItem("whichCondition"));
				}
			}
		} else alert("Check the box to certify you are finished with the task.");
	});
}

function drawScatterPlot(data) {

	// heterogeneous data
	// initdim1 = attr.indexOf(localStorage.getItem("defaultX_dog")), initdim2 = attr.indexOf(localStorage.getItem("defaultY_dog"));
	initdim1 = attr.indexOf(defaultX), initdim2 = attr.indexOf(defaultY); // 1, 2 = height, weight
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
		var pt_log = { credit: d.name, x: d.x, y: d.y, cx: +this.getAttribute("cx"), cy: +this.getAttribute("cy") };
		data_locations.push(pt_log);
	});

	//update IAL weight vector
	ial.usermodel.setAttributeWeightVector(V1, true, { 'level': 'INFO', 'taskType': 'credit', 'eventType': 'set_attribute_weight_vector_init', 'whichAxis': 'X', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'data_locations': data_locations });

	R7Insight.log(JSON.stringify(ial.logging.peek()));
	//LE.log(JSON.stringify(ial.logging.peek()));
	ial.usermodel.setAttributeWeightVector(V2, true, { 'level': 'INFO', 'taskType': 'credit', 'eventType': 'set_attribute_weight_vector_init', 'whichAxis': 'Y', 'userId': window.localStorage.getItem("userId"), 'whichCondition': window.localStorage.getItem("whichCondition"), 'data_locations': data_locations });

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