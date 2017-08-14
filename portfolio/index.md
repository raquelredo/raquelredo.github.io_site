---
layout: page
title: "Portfolio"
permalink: /portfolio/
author_profile: true
comments: false
---
The following projects have been done on my own or while doing some DS Courses. You can access each individual coding just by clicking any of them.

  {% for post in site.portfolio %}
    {% include archive-single.html type="grid" %}
  {% endfor %}

